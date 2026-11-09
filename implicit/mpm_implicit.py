# -*- coding: utf-8 -*-
"""
隐式 MPM 基准求解器（准静态，更新拉格朗日；载荷步内冻结粒子—网格映射，步末更新粒子与网格）。
参考：Guilkey & Weiss (2003)；Charlton, Coombs & Augarde (2017) iGIMP；Bonet & Wood 非线性连续介质力学。

Newton–Raphson：
  残量  R = f_ext − f_int(Δu)，f_int,i = Σ_p V_p⁰ P_rel ∇₀N_i，P_rel = J_rel σ F_rel^{-T}，F = F_rel F₀
  一致切线（空间形式）：K = Σ_p v_p [ (∇N_i)^T c (∇N_j) + (∇N_i·σ·∇N_j) I ]，∇N = F_rel^{-T} ∇₀N，v_p = J_rel V_p⁰
  Neo-Hookean 空间弹性张量：λ' = λ/J，μ' = (μ − λ lnJ)/J
线性求解器（solver）：
  'direct'  : 稀疏 LU（SuperLU，对称最小度排序 MMD_AT_PLUS_A；对应 MATLAB 反斜杠/chol）
  'pcg_ilu' : PCG + 不完全 LU（spilu；对应 MATLAB pcg + ilu）
  'pcg_2l'  : PCG + 两级预条件（ILU 平滑 + 粗空间 Φ 精确粗校正，对称乘性组合），粗空间见 implicit/precond.py
计时环节：shape, pattern, const, resid, tangent, precond, solve, update
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator
from common.shape import shape_2d
from common.material import neo_cauchy, neo_energy
from common.timer import Timer
from common.msbasis import extend_small_cut


def _pattern(idx, nn, free_map):
    """一个载荷步内固定的稀疏结构：COO→CSR 的累加映射（每次 Newton 迭代只需 bincount 求值）。"""
    Np = idx.shape[0]
    r = np.broadcast_to(idx[:, :, None], (Np, 9, 9)); c = np.broadcast_to(idx[:, None, :], (Np, 9, 9))
    R = np.concatenate([free_map[r], free_map[r], free_map[r + nn], free_map[r + nn]], axis=None)
    C = np.concatenate([free_map[c], free_map[c + nn], free_map[c], free_map[c + nn]], axis=None)
    keep = (R >= 0) & (C >= 0)
    nf = free_map.max() + 1
    key = R[keep].astype(np.int64) * nf + C[keep]
    ukey, inv = np.unique(key, return_inverse=True)
    rows = ukey // nf; cols = ukey % nf
    indptr = np.searchsorted(rows, np.arange(nf + 1))
    return dict(keep=keep, inv=inv, nnz=len(ukey), indptr=indptr, indices=cols.astype(np.int32), nf=nf)


def _tangent_vals(dX, dY, v, lam_, mu_, s11, s12, s22):
    """每粒子 4 个 9x9 块（xx, xy, yx, yy）的切线值。dX,dY 为当前构形梯度。"""
    V = v[:, None, None]; L = lam_[:, None, None]; M = mu_[:, None, None]
    gxi = dX[:, :, None]; gyi = dY[:, :, None]; gxj = dX[:, None, :]; gyj = dY[:, None, :]
    geo = V * (gxi * (s11[:, None, None] * gxj + s12[:, None, None] * gyj) +
               gyi * (s12[:, None, None] * gxj + s22[:, None, None] * gyj))
    Kxx = V * ((L + 2 * M) * gxi * gxj + M * gyi * gyj) + geo
    Kxy = V * (L * gxi * gyj + M * gyi * gxj)
    Kyx = V * (L * gyi * gxj + M * gxi * gyj)
    Kyy = V * ((L + 2 * M) * gyi * gyj + M * gxi * gxj) + geo
    return np.concatenate([Kxx, Kxy, Kyx, Kyy], axis=None)


def run_implicit(P, grid, par, solver='direct', precond_builder=None, verbose=True):
    """par: nload（载荷步数）、gravity（最终体力）、fixed_nodes、tol（相对残量）、maxit、
    lin_tol（PCG 相对容差）、reuse_prec（同一载荷步内复用预条件）。"""
    x = P['x'].copy(); m = P['m']; V0 = P['V0']; lam = P['lam']; mu = P['mu']
    Np = len(m); nn = grid['nn']
    F = [np.ones(Np), np.zeros(Np), np.zeros(Np), np.ones(Np)]
    gx, gy = par['gravity']; nload = par['nload']
    tol = par.get('tol', 1e-8); maxit = par.get('maxit', 30); lin_tol = par.get('lin_tol', 1e-10)
    T = Timer()
    hist = dict(step=[], newton=[], lin_it=[], res=[], tip=[], Es=[], ndof=[])
    for ls in range(1, nload + 1):
        fac = ls / nload
        # ---------- 载荷步初始化 ----------
        T.tic('shape')
        idx, N, dNx, dNy = shape_2d(x, grid, 'qbspline'); fi = idx.ravel()
        T.tic('pattern')
        mi = np.bincount(fi, (N * m[:, None]).ravel(), nn)
        act = np.zeros(2 * nn, bool); act[np.concatenate([idx.ravel(), idx.ravel() + nn])] = True
        act[par['fixed_nodes']] = False; act[np.asarray(par['fixed_nodes']) + nn] = False
        free_map = -np.ones(2 * nn, np.int64); fdofs = np.where(act)[0]; free_map[fdofs] = np.arange(len(fdofs))
        pat = _pattern(idx, nn, free_map)
        fext = np.concatenate([mi * gx * fac, mi * gy * fac])[fdofs]
        # 细尺度小支撑节点外推约束（扩展 B 样条）：du_free = E du_red，K_red = Eᵀ K E
        E = None
        if par.get('fine_extend', 0.0) > 0:
            gfine = dict(nc=nn, ncx=grid['nx'], ncy=grid['ny'])
            mh = np.concatenate([mi, mi]); mh[~act] = 0.0           # 固定/非活跃节点不参与（质量置零）
            Efull, einfo = extend_small_cut(sp.identity(2 * nn, format='csr'), gfine, mh, theta=par['fine_extend'])
            Ef = Efull[fdofs][:, fdofs].tocsc()
            keepc = np.where(np.asarray(abs(Ef).sum(0)).ravel() > 0)[0]
            E = Ef[:, keepc].tocsr(); ET = E.T.tocsr()
        du = np.zeros(2 * nn)
        F0 = [f.copy() for f in F]; Vp0 = V0 * (F0[0] * F0[3] - F0[1] * F0[2])
        prec = None; lin_total = 0; rn_prev = np.inf; n_ls = 0; step_len = 1.0; d_last = None
        for it in range(maxit + 1):
            # ---------- 本构（由当前位移增量） ----------
            T.tic('const')
            uxp = du[:nn][idx]; uyp = du[nn:][idx]
            D11 = np.sum(uxp * dNx, 1); D12 = np.sum(uxp * dNy, 1); D21 = np.sum(uyp * dNx, 1); D22 = np.sum(uyp * dNy, 1)
            a11 = 1 + D11; a12 = D12; a21 = D21; a22 = 1 + D22
            F = [a11 * F0[0] + a12 * F0[2], a11 * F0[1] + a12 * F0[3], a21 * F0[0] + a22 * F0[2], a21 * F0[1] + a22 * F0[3]]
            s11, s12, s22, J = neo_cauchy(*F, lam, mu)
            Jr = a11 * a22 - a12 * a21
            if not np.isfinite(J).all() or J.min() <= 0:
                T.stop(); raise FloatingPointError(f'载荷步 {ls} 迭代 {it}: J<=0')
            # ---------- 残量 ----------
            T.tic('resid')
            P11 = s11 * a22 - s12 * a12; P12 = -s11 * a21 + s12 * a11
            P21 = s12 * a22 - s22 * a12; P22 = -s12 * a21 + s22 * a11
            fx = np.bincount(fi, ((Vp0 * P11)[:, None] * dNx + (Vp0 * P12)[:, None] * dNy).ravel(), nn)
            fy = np.bincount(fi, ((Vp0 * P21)[:, None] * dNx + (Vp0 * P22)[:, None] * dNy).ravel(), nn)
            Rf = fext - np.concatenate([fx, fy])[fdofs]
            Rr = ET @ Rf if E is not None else Rf
            fr_ = ET @ fext if E is not None else fext
            rn = np.linalg.norm(Rr) / max(np.linalg.norm(fr_), 1e-30)
            # 回溯线搜索：残量增大时将上一步更新减半（最多 6 次）
            if par.get('linesearch', False) and it > 0 and rn > rn_prev and n_ls < 6:
                T.tic('linesearch')
                step_len *= 0.5; du[fdofs] -= step_len * d_last; n_ls += 1
                continue_ls = True
            else:
                continue_ls = False
            if continue_ls:
                continue
            rn_prev = rn; n_ls = 0
            if verbose:
                print(f'  载荷步 {ls:3d} 迭代 {it:2d}  |R|/|f| = {rn:.3e}', flush=True)
            if rn < tol:
                break
            if it == maxit:
                T.stop(); raise RuntimeError(f'载荷步 {ls} Newton 未收敛 |R|={rn:.2e}')
            # ---------- 一致切线 ----------
            T.tic('tangent')
            # 当前构形梯度 ∇N = F_rel^{-T} ∇₀N
            dX = (a22[:, None] * dNx - a21[:, None] * dNy) / Jr[:, None]
            dY = (-a12[:, None] * dNx + a11[:, None] * dNy) / Jr[:, None]
            lnJ = np.log(J)
            vals = _tangent_vals(dX, dY, Jr * Vp0, lam / J, (mu - lam * lnJ) / J, s11, s12, s22)
            data = np.bincount(pat['inv'], vals[pat['keep']], pat['nnz'])
            K = sp.csr_matrix((data, pat['indices'], pat['indptr']), shape=(pat['nf'], pat['nf']))
            if E is not None:
                K = (ET @ K @ E).tocsr()
            Rf_solve = Rr
            # ---------- 线性求解 ----------
            if solver == 'direct':
                T.tic('solve')
                lu = splu(K.tocsc(), permc_spec='MMD_AT_PLUS_A')
                d = lu.solve(Rf_solve); nl = 0
            else:
                if prec is None or not par.get('reuse_prec', False):
                    T.tic('precond')
                    prec = precond_builder(K, dict(x=x, Vp=Vp0, lam=lam, mu=mu, grid=grid, fdofs=fdofs, mi=mi,
                                                   fixed_nodes=par['fixed_nodes'], timer=T, E=E))
                T.tic('solve')
                cnt = [0]
                def cb(_): cnt[0] += 1
                d, info = cg(K, Rf_solve, rtol=lin_tol, atol=0.0, maxiter=par.get('cg_maxit', 2000), M=prec, callback=cb)
                nl = cnt[0]
                if info != 0:
                    print(f'    [警告] PCG 未在最大步数内收敛 info={info}')
            lin_total += nl
            if E is not None:
                d = E @ d
            du[fdofs] += d; d_last = d; step_len = 1.0
        # ---------- 载荷步末：更新粒子 ----------
        T.tic('update')
        x[:, 0] += np.sum(N * du[:nn][idx], 1); x[:, 1] += np.sum(N * du[nn:][idx], 1)
        Es = np.sum(V0 * neo_energy(*F, lam, mu))
        hist['step'].append(ls); hist['newton'].append(it); hist['lin_it'].append(lin_total); hist['res'].append(rn)
        hist['tip'].append(x[par['monitor']].ravel().copy()); hist['Es'].append(Es); hist['ndof'].append(E.shape[1] if E is not None else pat['nf'])
        if verbose:
            print(f'[载荷步 {ls}/{nload}] Newton {it} 次，线性迭代 {lin_total}，自由度 {hist["ndof"][-1]}，'
                  f'自由端 y = {x[par["monitor"][0], 1]:.5f}', flush=True)
    T.stop()
    s11, s12, s22, J = neo_cauchy(*F, lam, mu)
    return dict(x=x, F=np.stack(F, 1), s=np.stack([s11, s12, s22], 1), hist={k: np.array(v) for k, v in hist.items()},
                timer=T, Np=Np)
