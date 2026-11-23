# -*- coding: utf-8 -*-
"""
隐式 MPM：多尺度两级区域分解预条件的非精确修正 Newton（I3c，非学习加速；可接 PIML 粗空间）。

求解的离散问题与细尺度基准完全相同（同一粒子积分、同一细尺度小支撑外推 E、同一收敛准则 |R|/|f| < tol），
因此结果收敛到细尺度解，精度由容差控制（区别于"纯粗空间约化"）。

每载荷步（预条件器构造，计时 pc_*）：
  K0      ：载荷步起点的一致切线（细尺度，一次）
  粗空间 P ：边自适应多尺度基 + 小切割外推 + 有限转动缺陷增强（basis='os'；或 'os_piml' 用 PIML 边条带）
  局部块   ：以每个粗节点为中心、覆盖相邻 2×2 粗单元的细自由度（重叠），对 K0 子矩阵做稀疏 LU
  粗算子   ：Kc = Pᵀ K0 P（稀疏 LU）
每次 Newton 迭代：
  残量 R（精确，全部粒子）→ 以 K0 为系统矩阵、两级对称加性 Schwarz 为预条件的 PCG（相对容差 lin_tol）求修正量。
  （修正 Newton：不每次重组切线；残量精确 ⇒ 收敛到细尺度解。）
计时环节：shape, pattern, const, resid, tangent, pc_basis, pc_patch, pc_coarse, pcg, update
目标模块 = pattern + tangent + pc_* + pcg（与基准的 pattern + tangent + solve 对应）
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator
from common.shape import shape_2d
from common.material import neo_cauchy, neo_energy
from common.timer import Timer
from common.msbasis import extend_small_cut, coarse_global
from implicit.mpm_implicit import _pattern_struct, assemble_tangent


def coarse_space_dd(x, Vp, lam, mu, grid, n, fixed_nodes, mi, fdofs, Ered_rows, basis='os', edge_models=None, theta=0.1):
    from common.msbasis_os import build_basis_os
    nn = grid['nn']
    B = build_basis_os(x, Vp, lam, mu, grid, n, fixed_nodes=fixed_nodes, edge_models=edge_models)
    Phi, _ = extend_small_cut(B['Phi'], B['cg'], np.concatenate([mi, mi]), theta)
    xc = B['cg']['xc']; xn = grid['xn']
    ev = Phi @ np.concatenate([xc[:, 0], xc[:, 1]]); ex = ev[:nn] - xn[:, 0]; ey = ev[nn:] - xn[:, 1]
    rs = np.asarray(abs(Phi).sum(1)).ravel(); dead = (rs[:nn] == 0) & (rs[nn:] == 0); ex[dead] = 0; ey[dead] = 0
    Cq = coarse_global(grid, B['cg'], np.arange(nn), 'qbspline')
    Phi = sp.hstack([Phi, sp.vstack([sp.diags(ex) @ Cq, sp.diags(ey) @ Cq])]).tocsr()
    P = Phi[fdofs][Ered_rows].tocsc()
    cn = np.sqrt(np.asarray(P.multiply(P).sum(0)).ravel())
    return P[:, cn > 1e-8 * cn.max()].tocsr(), B['cg']


def patches_dd(dof_nodes, grid, n):
    """以粗节点为中心的重叠块：返回每块的自由度索引（相对约化自由度编号）。"""
    nx = grid['nx']; ix = dof_nodes % nx; iy = dof_nodes // nx
    A = np.floor((ix - 0.5) / n).astype(np.int64); Bc = np.floor((iy - 0.5) / n).astype(np.int64)
    # 每个自由度属于 4 个块：(A,B),(A+1,B),(A,B+1),(A+1,B+1)（块以粗节点 (a,b) 命名，覆盖单元 a-1..a × b-1..b）
    keys = []; ids = []
    base = np.arange(len(dof_nodes))
    for da in (0, 1):
        for db in (0, 1):
            keys.append((A + da) * 100003 + (Bc + db)); ids.append(base)
    keys = np.concatenate(keys); ids = np.concatenate(ids)
    order = np.argsort(keys, kind='stable'); keys = keys[order]; ids = ids[order]
    cut = np.flatnonzero(np.diff(keys)) + 1
    return np.split(ids, cut)


def run_implicit_dd(P, grid, par, n=8, basis='os', edge_models=None, theta=0.1, verbose=False):
    x = P['x'].copy(); m = P['m']; V0 = P['V0']; lam = P['lam']; mu = P['mu']
    Np = len(m); nn = grid['nn']
    F = [np.ones(Np), np.zeros(Np), np.zeros(Np), np.ones(Np)]
    gx, gy = par['gravity']; nload = par['nload']; tol = par.get('tol', 1e-8); maxit = par.get('maxit', 60)
    lin_tol = par.get('lin_tol', 1e-2)
    T = Timer(); hist = dict(step=[], newton=[], lin_it=[], res=[], tip=[], Es=[], ndof=[], npatch=[], pc_built=[])
    pc_every = par.get('pc_every', 1)          # 预条件器每 pc_every 个载荷步重建一次（其余步复用，按全局自由度编号映射）
    PC = None
    for ls in range(1, min(nload, par.get('stop_after', nload)) + 1):
        fac = ls / nload
        T.tic('shape')
        idx, N, dNx, dNy = shape_2d(x, grid, 'qbspline'); fi = idx.ravel()
        T.tic('pattern')
        mi = np.bincount(fi, (N * m[:, None]).ravel(), nn)
        act = np.zeros(2 * nn, bool); act[np.concatenate([fi, fi + nn])] = True
        act[par['fixed_nodes']] = False; act[np.asarray(par['fixed_nodes']) + nn] = False
        free_map = -np.ones(2 * nn, np.int64); fdofs = np.where(act)[0]; free_map[fdofs] = np.arange(len(fdofs))
        pat = _pattern_struct(idx, nn, free_map, grid['nx'])
        fext = np.concatenate([mi * gx * fac, mi * gy * fac])[fdofs]
        # 细尺度小支撑外推约束（与基准相同）
        gfine = dict(nc=nn, ncx=grid['nx'], ncy=grid['ny'])
        mh = np.concatenate([mi, mi]); mh[~act] = 0.0
        Efull, _ = extend_small_cut(sp.identity(2 * nn, format='csr'), gfine, mh, theta=par.get('fine_extend', 0.05))
        Ef = Efull[fdofs][:, fdofs].tocsc()
        keepc = np.where(np.asarray(abs(Ef).sum(0)).ravel() > 0)[0]
        E = Ef[:, keepc].tocsr(); ET = E.T.tocsr()
        good_rows = np.asarray(E.tocsc().argmax(0)).ravel()           # 约化自由度对应的自由度行（好节点处 E 为单位阵）
        du = np.zeros(2 * nn)
        F0 = [f.copy() for f in F]; Vp0 = V0 * (F0[0] * F0[3] - F0[1] * F0[2])
        fr_ = ET @ fext; nf_ = max(np.linalg.norm(fr_), 1e-30)
        lin_total = 0; K0 = None

        def state(du_):
            uxp = du_[:nn][idx]; uyp = du_[nn:][idx]
            a11 = 1 + np.sum(uxp * dNx, 1); a12 = np.sum(uxp * dNy, 1); a21 = np.sum(uyp * dNx, 1); a22 = 1 + np.sum(uyp * dNy, 1)
            F_ = [a11 * F0[0] + a12 * F0[2], a11 * F0[1] + a12 * F0[3], a21 * F0[0] + a22 * F0[2], a21 * F0[1] + a22 * F0[3]]
            s11, s12, s22, J = neo_cauchy(*F_, lam, mu)
            return F_, (a11, a12, a21, a22), (s11, s12, s22), J

        def fint_red(a, s_):
            a11, a12, a21, a22 = a; s11, s12, s22 = s_
            P11 = s11 * a22 - s12 * a12; P12 = -s11 * a21 + s12 * a11; P21 = s12 * a22 - s22 * a12; P22 = -s12 * a21 + s22 * a11
            fx = np.bincount(fi, ((Vp0 * P11)[:, None] * dNx + (Vp0 * P12)[:, None] * dNy).ravel(), nn)
            fy = np.bincount(fi, ((Vp0 * P21)[:, None] * dNx + (Vp0 * P22)[:, None] * dNy).ravel(), nn)
            return ET @ np.concatenate([fx, fy])[fdofs]

        jv_mode = par.get('jv', 'fd')
        for it in range(maxit + 1):
            T.tic('const')
            F, aa, ss, J = state(du)
            a11, a12, a21, a22 = aa; s11, s12, s22 = ss
            Jr = a11 * a22 - a12 * a21
            if not np.isfinite(J).all() or J.min() <= 0:
                T.stop(); raise FloatingPointError(f'载荷步 {ls} 迭代 {it}: J<=0')
            T.tic('resid')
            fint0 = fint_red(aa, ss)
            Rr = ET @ fext - fint0
            rn = np.linalg.norm(Rr) / nf_
            if verbose:
                print(f'  载荷步 {ls} 迭代 {it} |R|/|f|={rn:.2e}', flush=True)
            if rn < tol:
                break
            if it == maxit:
                T.stop(); raise RuntimeError(f'载荷步 {ls} 未收敛 {rn:.2e}')
            gids = fdofs[good_rows]                                       # 当前约化未知量的全局自由度编号
            if K0 is None and PC is not None and (ls - PC['ls']) % pc_every != 0:
                # ---------- 复用上次构造的预条件器（不组装切线、不重建基与局部分解） ----------
                T.tic('pc_map')
                pos = np.searchsorted(PC['gids'], gids); pos = np.clip(pos, 0, len(PC['gids']) - 1)
                ok = PC['gids'][pos] == gids; posok = pos[ok]; nold = len(PC['gids']); dsc = PC['dmean']

                def Mapply_map(r, posok=posok, ok=ok, nold=nold, dsc=dsc):
                    ro = np.zeros(nold); ro[posok] = r[ok]
                    zo = PC['apply'](ro)
                    z = r / dsc; z[ok] = zo[posok]
                    return z
                Mop = LinearOperator((len(gids), len(gids)), matvec=Mapply_map, dtype=float)
                K0 = 'reused'
            if K0 is None:
                # ---------- 预条件器构造（每 pc_every 个载荷步一次） ----------
                T.tic('tangent')
                dX = (a22[:, None] * dNx - a21[:, None] * dNy) / Jr[:, None]; dY = (-a12[:, None] * dNx + a11[:, None] * dNy) / Jr[:, None]
                K = assemble_tangent(pat, dX, dY, Jr * Vp0, lam / J, (mu - lam * np.log(J)) / J, s11, s12, s22)
                K0 = (ET @ K @ E).tocsr()
                T.tic('pc_basis')
                Pc, cg_ = coarse_space_dd(x, Vp0, lam, mu, grid, n, par['fixed_nodes'], mi, fdofs, good_rows,
                                          basis=basis, edge_models=edge_models)
                T.tic('pc_patch')
                dof_nodes = fdofs[good_rows] % nn
                blocks = patches_dd(dof_nodes, grid, n)
                K0c = K0.tocsc()
                lus = [(b, splu(K0c[b][:, b].tocsc(), permc_spec='MMD_AT_PLUS_A')) for b in blocks]
                T.tic('pc_coarse')
                clu = splu((Pc.T @ K0 @ Pc).tocsc(), permc_spec='MMD_AT_PLUS_A'); PcT = Pc.T.tocsr()

                def Mapply(r):
                    z = np.zeros_like(r)
                    for b, lu in lus:
                        z[b] += lu.solve(r[b])
                    return z + Pc @ clu.solve(PcT @ r)
                Mop = LinearOperator(K0.shape, matvec=Mapply, dtype=float)
                PC = dict(ls=ls, gids=gids.copy(), apply=Mapply, dmean=float(np.mean(K0.diagonal())), npatch=len(blocks))
                hist['pc_built'].append(ls)
            if jv_mode == 'assemble':          # 每次迭代组装当前切线（完整 Newton，预条件器仍来自 K0）
                T.tic('tangent')
                dX = (a22[:, None] * dNx - a21[:, None] * dNy) / Jr[:, None]; dY = (-a12[:, None] * dNx + a11[:, None] * dNy) / Jr[:, None]
                Kit = (ET @ assemble_tangent(pat, dX, dY, Jr * Vp0, lam / J, (mu - lam * np.log(J)) / J, s11, s12, s22) @ E).tocsr()
                Aop = Kit
            elif jv_mode == 'fd':              # Jacobian-free：J v ≈ [f_int(u + εv) − f_int(u)]/ε（每次一残量求值）
                unorm = np.linalg.norm(du[fdofs])

                def jv(v):
                    T.tic('pcg_jv')
                    vn = np.linalg.norm(v)
                    if vn == 0:
                        return np.zeros_like(v)
                    eps = np.sqrt(np.finfo(float).eps) * (1.0 + unorm) / vn
                    dup = du.copy(); dup[fdofs] += eps * (E @ v)
                    _, a_, s_, _ = state(dup)
                    out = (fint_red(a_, s_) - fint0) / eps
                    T.tic('pcg')
                    return out
                Aop = LinearOperator((len(Rr), len(Rr)), matvec=jv, dtype=float)
            else:                              # 'K0'：修正 Newton（冻结切线；仅在本步构造了 K0 时可用）
                Aop = K0
            T.tic('pcg')
            cnt = [0]
            d, info = cg(Aop, Rr, rtol=lin_tol, atol=0.0, maxiter=500, M=Mop, callback=lambda _: cnt.__setitem__(0, cnt[0] + 1))
            lin_total += cnt[0]
            du[fdofs] += E @ d
        T.tic('update')
        x[:, 0] += np.sum(N * du[:nn][idx], 1); x[:, 1] += np.sum(N * du[nn:][idx], 1)
        hist['step'].append(ls); hist['newton'].append(it); hist['lin_it'].append(lin_total); hist['res'].append(rn)
        hist['tip'].append(x[par['monitor']].ravel().copy()); hist['Es'].append(float(np.sum(V0 * neo_energy(*F, lam, mu))))
        hist['ndof'].append(E.shape[1]); hist['npatch'].append(PC['npatch'] if PC else 0)
        if verbose:
            print(f'[载荷步 {ls}] Newton {it}，PCG {lin_total}，预条件器构造于第 {PC["ls"]} 步', flush=True)
    T.stop()
    s11, s12, s22, J = neo_cauchy(*F, lam, mu)
    return dict(x=x, F=np.stack(F, 1), s=np.stack([s11, s12, s22], 1), hist={k: np.array(v) for k, v in hist.items()},
                timer=T, Np=Np)
