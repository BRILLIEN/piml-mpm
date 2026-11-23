# -*- coding: utf-8 -*-
"""
隐式 Newmark（平均加速度）一个时间步：与用户 MATLAB implicit_step_heterogeneous.m 的流程一致
（P2G → 隐式 Newton（逐轮一致切线 + 能量线搜索）→ G2P → 粒子更新）。

求解模式 mode：
  'fine_pcg'    细解：完整活动细自由度上的 Jacobi-PCG（相对容差 3e-10）
  'sub_current' 子结构（用户方案）：三线性粗界面 + 两层共享细界面，逐轮以当轮有效切线做块内凝聚，粗界面方程直接分解求解
  'subpc'       子结构预条件细空间 PCG（本研究新增）：在完整细空间上用当轮一致切线 K_eff 作 PCG，
                预条件器 = 子结构凝聚算子 E(EᵀK E)⁻¹Eᵀ（与 sub_current 同一套块内分解与粗界面方程）+ 界面细自由度 Jacobi。
                结果收敛到细解（同一容差），精度与细解一致；代价 = 子结构构造 + 少量细空间矩阵向量积。
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, LinearOperator
from implicit3d.beam_kernels import (beam_basis, beam_transfer, beam_neo, beam_tangent_pattern, beam_tangent,
                                     BasicMat_trace, overlap_patches)


def solve_pcg_beam(A, diagonal, b, tol=3e-10, maxit=20000):
    """同一 Jacobi-PCG 同时用于完整方程（与 MATLAB solve_pcg_beam 一致）。"""
    assert np.all(diagonal > 0), 'PCG requires positive diagonal'
    it = [0]
    u, info = cg(A, b, rtol=tol, atol=0.0, maxiter=maxit, M=LinearOperator(A.shape, matvec=lambda z: z / diagonal),
                 callback=lambda _: it.__setitem__(0, it[0] + 1))
    return u, it[0]


class Condense:
    """块内精确凝聚到三线性粗界面（逐轮当轮有效切线）。apply(r) = E (EᵀKE)⁻¹ Eᵀ r。"""

    def __init__(self, K, di, db, Nb, blockID, timer):
        t = time.perf_counter()
        self.di, self.db, self.Nb = di, db, Nb.tocsr()
        Kc = K.tocsr()
        Kib = Kc[di][:, db] @ self.Nb                                  # (ni_tot, ncoarse) = C
        self.groups = []
        order = np.argsort(blockID, kind='stable'); lab = blockID[order]
        starts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, lab.size]
        sizes = np.diff(starts)
        timer['local_extract'] += time.perf_counter() - t
        Scoo_r, Scoo_c, Scoo_v = [], [], []
        for ni in np.unique(sizes):
            t = time.perf_counter()
            bl = np.where(sizes == ni)[0]; nbk = bl.size
            rows = order[starts[bl][:, None] + np.arange(ni)[None, :]]              # (nbk, ni) 相对 di 的位置
            dof = di[rows]                                                          # 活动自由度编号
            Kii = np.empty((nbk, ni, ni))
            for j in range(nbk):
                Kii[j] = Kc[dof[j]][:, dof[j]].toarray()
            Cs = [Kib[rows[j]] for j in range(nbk)]
            cols = [np.unique(c.indices) for c in Cs]
            ncol = max(len(c) for c in cols)
            colpad = np.zeros((nbk, ncol), np.int64); cmask = np.zeros((nbk, ncol), bool)
            C = np.zeros((nbk, ni, ncol))
            for j in range(nbk):
                colpad[j, :len(cols[j])] = cols[j]; cmask[j, :len(cols[j])] = True
                C[j, :, :len(cols[j])] = Cs[j][:, cols[j]].toarray()
            timer['local_extract'] += time.perf_counter() - t
            t = time.perf_counter()
            Kinv = np.linalg.inv(Kii)                                               # 块内（81×81）稠密批量求逆
            timer['local_factor'] += time.perf_counter() - t
            t = time.perf_counter()
            H = Kinv @ C                                                            # 块内响应（数值形函数）
            SC = np.einsum('bic,bid->bcd', C, H)                                    # Cᵀ Kii⁻¹ C
            timer['local_response'] += time.perf_counter() - t
            t = time.perf_counter()
            m = cmask[:, :, None] & cmask[:, None, :]
            Scoo_r.append(np.broadcast_to(colpad[:, :, None], SC.shape)[m])
            Scoo_c.append(np.broadcast_to(colpad[:, None, :], SC.shape)[m]); Scoo_v.append(-SC[m])
            self.groups.append(dict(rows=rows, Kinv=Kinv, C=C, H=H, colpad=colpad, cmask=cmask))
            timer['condense'] += time.perf_counter() - t
        t = time.perf_counter()
        Kbb = Kc[db][:, db]
        S = (self.Nb.T @ Kbb @ self.Nb).tocoo()
        nc = S.shape[0]
        S = sp.csr_matrix((np.r_[S.data, np.concatenate(Scoo_v)], (np.r_[S.row, np.concatenate(Scoo_r)],
                                                                    np.r_[S.col, np.concatenate(Scoo_c)])), shape=(nc, nc))
        self.S = S
        timer['condense'] += time.perf_counter() - t
        t = time.perf_counter()
        self.Slu = splu(S.tocsc(), permc_spec='MMD_AT_PLUS_A')
        timer['coarse_solve'] += time.perf_counter() - t
        self.timer = timer
        self.NbT = self.Nb.T.tocsr()

    def apply(self, r):
        """E (EᵀKE)⁻¹ Eᵀ r：先块内消元得到粗界面右端，求粗界面，再块内回代。"""
        T = self.timer; t = time.perf_counter()
        ri = r[self.di]
        rhs = self.NbT @ r[self.db]
        Ys = []
        for g in self.groups:
            y = np.einsum('bij,bj->bi', g['Kinv'], ri[g['rows']])                    # Kii⁻¹ r_i
            f = np.einsum('bic,bi->bc', g['C'], y)                                   # Cᵀ Kii⁻¹ r_i
            np.subtract.at(rhs, g['colpad'][g['cmask']], f[g['cmask']])
            Ys.append(y)
        T['condense'] += time.perf_counter() - t; t = time.perf_counter()
        dq = self.Slu.solve(rhs)
        T['coarse_solve'] += time.perf_counter() - t; t = time.perf_counter()
        out = np.zeros_like(r)
        out[self.db] = self.Nb @ dq
        ui = np.zeros(self.di.size)
        for g, y in zip(self.groups, Ys):
            dqb = np.where(g['cmask'], dq[g['colpad']], 0.0)
            ui[g['rows']] = y - np.einsum('bic,bc->bi', g['H'], dqb)
        out[self.di] = ui
        T['recover'] += time.perf_counter() - t
        return out


def implicit_step_heterogeneous(s, cache, grid, p, mode, step, show):
    tstep = time.perf_counter()
    timer = {k: 0.0 for k in ['resid_const', 'tangent', 'pattern', 'trace', 'local_extract', 'local_factor', 'local_response',
                              'condense', 'coarse_solve', 'recover', 'pcg', 'linesearch']}
    h = p['h']; nf = grid['node'].shape[0]
    # ---------------- P2G ----------------
    t = time.perf_counter()
    patch, signature = beam_basis(s, grid, p)
    mass, momentum = beam_transfer('p2g', patch, grid, s, h=h)
    active = np.where((mass > p['activeTol'] * mass.max()) & ~grid['fixed'])[0]
    free = (3 * active[:, None] + np.arange(3)).ravel()
    grid_v = np.zeros((3, nf)); grid_v[:, active] = momentum[:, active] / mass[active]
    inertia = np.repeat((4 / p['dt'] ** 2) * mass[active], 3)
    t_p2g = time.perf_counter() - t
    # ---------------- 隐式 ----------------
    ti = time.perf_counter()
    U = np.zeros(3 * nf)
    t = time.perf_counter()
    P0, _, _, _, _, _ = beam_neo(s['F'], grid)
    f0 = beam_transfer('force', patch, grid, s, P0, h=h)
    fgravity = np.outer(p['gravity'], mass[active]).ravel(order='F')
    drive = p['dt'] * inertia * grid_v[:, active].ravel(order='F') - f0[:, active].ravel(order='F') + 2 * fgravity
    timer['resid_const'] += time.perf_counter() - t
    # 跨载荷步复用一致切线（名称含 _reuseM：每 M 步重建；活动集变化时提前重建）；残余力始终精确
    reuseM = int(mode.split('_reuse')[1].split('_')[0]) if '_reuse' in mode else 0
    reuse_ok = (reuseM > 0 and cache.get('Ktan_step') is not None and (step - cache['Ktan_step']) < reuseM
                and np.array_equal(cache.get('active'), active))
    t = time.perf_counter()
    if reuse_ok:
        pat = cache['pat']
    else:
        pat = beam_tangent_pattern(grid, active)
    timer['pattern'] += time.perf_counter() - t
    def trace(shift):
        db_, di_, Nb_, bn_, in_ = BasicMat_trace(grid['node'], active, grid['fixed'], grid['dims'], p['fine'], shift)
        own = (grid['node'][active[in_]] - shift) // p['fine']
        nblk = np.ceil(grid['dims'] / p['fine']).astype(np.int64) + 1
        return db_, di_, Nb_, np.repeat(own[:, 0] + nblk[0] * (own[:, 1] + nblk[1] * own[:, 2]), 3)
    import re as _re
    base_mode = _re.sub(r'_reuse\d+', '', mode).replace('_loose', '').replace('_frozen', '')
    frozen = '_frozen' in mode or '_reuse' in mode   # 步内冻结一致切线（每步至多组装一次；残余力始终精确）
    K = None
    if base_mode in ('sub_current', 'subpc', 'subpc2', 'subpc3'):
        t = time.perf_counter()
        db, di, Nb, blockID = trace(0)
        if base_mode == 'subpc2':
            db2, di2, Nb2, blockID2 = trace(2)            # 错位 2 个细节点的第二分区：原界面两层落入其块内部
        timer['trace'] += time.perf_counter() - t
    res = []; nit_total = 0
    for it in range(41):
        t = time.perf_counter()
        F = s['F'] + beam_transfer('increment', patch, grid, s, U.reshape(3, -1, order='F'), h=h)
        P, T, gamma, J, W, sigma = beam_neo(F, grid)
        assert P is not None, 'Invalid trial deformation'
        fint = beam_transfer('force', patch, grid, s, P, h=h)
        r = drive - fint[:, active].ravel(order='F') - inertia * U.reshape(3, -1, order='F')[:, active].ravel(order='F')
        fullNorm = np.linalg.norm(r)
        if base_mode == 'sub_current':
            f = np.r_[r[di], Nb.T @ r[db]]; solveNorm = np.linalg.norm(f)
        else:
            solveNorm = fullNorm
        if it == 0:
            ref = max(solveNorm, 1e-300); fullRef = max(fullNorm, 1e-300)
        res.append((it, solveNorm, solveNorm / ref, fullNorm, fullNorm / fullRef))
        timer['resid_const'] += time.perf_counter() - t
        ntol = p.get('looseTol', p['newtonTol']) if '_loose' in mode else p['newtonTol']   # 精度受控（宽松 Newton 容差）变体
        if solveNorm <= max(ntol * ref, p['absTol']):
            break
        assert it < 40, f'Newton stagnation {mode} step {step}'
        # ---------- 当轮一致切线 ----------
        if K is None and reuse_ok:
            t = time.perf_counter()
            K = cache['Ktan'] + sp.diags(inertia)                  # 复用旧切线 + 当前步惯性项
            timer['tangent'] += time.perf_counter() - t
        if K is None or not frozen:
            t = time.perf_counter()
            K = beam_tangent(patch, T, gamma, grid, pat, inertia)
            timer['tangent'] += time.perf_counter() - t
            if reuseM > 0:
                cache.update(Ktan=sp.csr_matrix((pat['last_data'], pat['indices'], pat['indptr']), shape=K.shape),
                             Ktan_step=step, active=active.copy(), pat=pat)
                tangentBuilds_flag = True
        if base_mode in ('fine_pcg', 'fine_pcg_inexact'):
            t = time.perf_counter()
            df, nit = solve_pcg_beam(K, K.diagonal(), r, tol=(p.get('subpc_tol', 3e-10) if base_mode == 'fine_pcg_inexact' else 3e-10))
            timer['pcg'] += time.perf_counter() - t
        else:
            cond = Condense(K, di, db, Nb, blockID, timer)
            if base_mode == 'sub_current':
                df = cond.apply(r); nit = 0
            else:
                if base_mode == 'subpc3':                      # 子结构凝聚（粗）+ 重叠 3×3×3 节点块精确局部解（加性两级 Schwarz）
                    t = time.perf_counter()
                    OP = overlap_patches(grid, active, pat, inertia)
                    timer['local_factor'] += time.perf_counter() - t
                    dofp = OP['dof']; vm = dofp >= 0; dsafe = np.maximum(dofp, 0); Kinv = OP['Kinv']

                    def Mfun(z):
                        t0 = time.perf_counter()
                        zz = np.where(vm, z[dsafe], 0.0)
                        y = np.einsum('pij,pj->pi', Kinv, zz)
                        out = np.bincount(dsafe[vm], y[vm], z.size)
                        timer['recover'] += time.perf_counter() - t0
                        return out + cond.apply(z)
                    M = LinearOperator(K.shape, matvec=Mfun)
                elif base_mode == 'subpc2':                      # 两个错位分区的子结构凝聚算子加性组合（对称正定）
                    cond2 = Condense(K, di2, db2, Nb2, blockID2, timer)
                    # 三维中"在两个分区里都不是块内部"的节点（如 x 位于界面层且 y 位于另一分区界面层）只受两个三线性界面约束，
                    # 加性组合在其上奇异 ⇒ 对这些自由度再加 Jacobi 项，保证预条件器对称正定。
                    unc = np.ones(r.size, bool); unc[di] = False; unc[di2] = False
                    Du = np.where(unc, 1.0 / K.diagonal(), 0.0)
                    M = LinearOperator(K.shape, matvec=lambda z: cond.apply(z) + cond2.apply(z) + Du * z)
                else:
                    Db = np.zeros(r.size); Db[db] = 1.0 / K.diagonal()[db]
                    w = p.get('omega', 1.0)
                    M = LinearOperator(K.shape, matvec=lambda z: cond.apply(z) + w * Db * z)
                t = time.perf_counter()
                cnt = [0]
                df, info = cg(K, r, rtol=p.get('subpc_tol', 3e-10), atol=0.0, maxiter=5000, M=M,
                              callback=lambda _: cnt.__setitem__(0, cnt[0] + 1))
                nit = cnt[0]
                timer['pcg'] += time.perf_counter() - t
        nit_total += nit
        # ---------- 能量线搜索（与 MATLAB 一致） ----------
        t = time.perf_counter()
        du = np.zeros(3 * nf).reshape(3, -1, order='F'); du[:, active] = df.reshape(3, -1, order='F'); du = du.ravel(order='F')
        uf = U.reshape(3, -1, order='F')[:, active].ravel(order='F')
        energy = np.sum(grid['V'] * W) + .5 * np.sum(inertia * uf ** 2) - drive @ uf
        slope = r @ df; alpha = 1.0; accepted = False
        for trial in range(18):
            cand = U + alpha * du
            Ft = s['F'] + beam_transfer('increment', patch, grid, s, cand.reshape(3, -1, order='F'), h=h)
            _, _, _, Jt, Wt, _ = beam_neo(Ft, grid, True)
            cf = cand.reshape(3, -1, order='F')[:, active].ravel(order='F')
            newE = np.sum(grid['V'] * Wt) + .5 * np.sum(inertia * cf ** 2) - drive @ cf
            if np.all(Jt > 1e-8) and newE <= energy - 1e-4 * alpha * slope + 1e-13 * max(abs(energy), 1):
                accepted = True; break
            alpha /= 2
        assert accepted, f'Line search failed {mode} step {step}'
        U = cand
        timer['linesearch'] += time.perf_counter() - t
    t_imp = time.perf_counter() - ti
    # ---------------- G2P 与粒子更新 ----------------
    t = time.perf_counter()
    dx, v, C = beam_transfer('g2p', patch, grid, s, U.reshape(3, -1, order='F'),
                             (2 * U / p['dt'] - grid_v.ravel(order='F')).reshape(3, -1, order='F'), h=h)
    t_g2p = time.perf_counter() - t
    t = time.perf_counter()
    s['x'] = s['x'] + dx; s['v'] = v; s['C'] = C; s['F'] = F; s['W'] = W; s['sig'] = sigma
    t_upd = time.perf_counter() - t
    row = dict(step=step, nfree=free.size, nglobal=(Nb.shape[1] if mode.startswith('sub') else free.size),
               newton=len(res) - 1, pcg_it=nit_total, p2g=t_p2g, implicit=t_imp, g2p=t_g2p, update=t_upd,
               total=time.perf_counter() - tstep, res=res, **timer)
    if show:
        print(f"[MPM {mode}] 步 {step} | DOF {free.size} | P2G {t_p2g:.3f} | 隐式 {t_imp:.3f} | G2P {t_g2p:.3f} | 粒子更新 {t_upd:.3f} | 本步 {row['total']:.3f} s")
        print(f"  [隐式] 切线与残量 {timer['tangent'] + timer['resid_const'] + timer['pattern']:.3f} | 局部响应 "
              f"{timer['local_extract'] + timer['local_factor'] + timer['local_response']:.3f} | 凝聚装配 {timer['condense'] + timer['trace']:.3f} | "
              f"迭代求解 {timer['pcg'] + timer['coarse_solve']:.3f} | 恢复 {timer['recover']:.3f} | 线搜索 {timer['linesearch']:.3f} s")
        space = '约束空间' if mode == 'sub_current' else '完整细空间'
        print(f"  Newton {row['newton']} 次 | PCG {nit_total} 次（残余力空间：{space}）")
        for q in res:
            print(f"    修正后 {q[0]:02d} | 求解空间残余力 {q[1]:.6e}（相对 {q[2]:.3e}） | 原细空间残余力 {q[3]:.6e}（相对 {q[4]:.3e}）")
    return s, cache, row
