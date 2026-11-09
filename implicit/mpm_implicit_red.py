# -*- coding: utf-8 -*-
"""
隐式多尺度约化 Newton（非学习加速 I3b / PIML I4）：
  机制 M2（降自由度）：载荷步内位移增量限制在多尺度粗空间 du = Φ U，求解 K_c = Φᵀ K Φ（规模 ≈ 细自由度/n²）。
  机制 M3（降算子求值成本）：残量与切线在"粒子积分聚合点"上计算（同一细区间同一相的粒子合并，见显式 §4.8），
       Nq ≈ 0.3 N_p；粗刚度满秩保证不出现沙漏模态。
  机制 PIML：Φ 的边条带由网络预测（ml/train_edge.py），含固定节点的边回退精确。
与细尺度基准相同：UL 准静态、载荷步内冻结映射、一致切线（材料 + 几何）、Neo-Hookean、相同收敛准则（相对残量在粗空间度量）。
区间（载荷步）末：全体粒子按 du = Φ U 精确更新位置与变形梯度。
计时：shape, basis, agg, pattern, const, resid, tangent, project, solve, update
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from common.shape import shape_2d
from common.material import neo_cauchy, neo_energy
from common.timer import Timer
from common.msbasis import extend_small_cut
from implicit.mpm_implicit import _pattern, _tangent_vals


def run_implicit_reduced(P, grid, par, n=8, basis='os', agg=1, edge_models=None, theta=0.1, verbose=False):
    from common.msbasis_os import build_basis_os
    from common.msbasis import build_basis
    x = P['x'].copy(); m = P['m']; V0 = P['V0']; lam = P['lam']; mu = P['mu']
    Np = len(m); nn = grid['nn']
    F = [np.ones(Np), np.zeros(Np), np.zeros(Np), np.ones(Np)]
    gx, gy = par['gravity']; nload = par['nload']; tol = par.get('tol', 1e-8); maxit = par.get('maxit', 30)
    _, mat_id = np.unique(np.stack([lam, mu], 1), axis=0, return_inverse=True); mat_id = mat_id.ravel()
    T = Timer(); hist = dict(newton=[], ndof=[], nq=[], tip=[])
    for ls in range(1, nload + 1):
        fac = ls / nload
        T.tic('shape')
        idx, N, dNx, dNy = shape_2d(x, grid, 'qbspline'); fi = idx.ravel()
        mi = np.bincount(fi, (N * m[:, None]).ravel(), nn)
        F0 = [f.copy() for f in F]; Vp0 = V0 * (F0[0] * F0[3] - F0[1] * F0[2])
        # ---------- 多尺度基（状态相关，每载荷步一次） ----------
        T.tic('basis')
        if basis == 'ms':
            B = build_basis(x, Vp0, lam, mu, grid, n, fixed_nodes=par['fixed_nodes'])
        else:
            B = build_basis_os(x, Vp0, lam, mu, grid, n, fixed_nodes=par['fixed_nodes'], edge_models=edge_models)
        Phi, _ = extend_small_cut(B['Phi'], B['cg'], np.concatenate([mi, mi]), theta)
        act = np.zeros(2 * nn, bool); act[np.concatenate([fi, fi + nn])] = True
        act[par['fixed_nodes']] = False; act[np.asarray(par['fixed_nodes']) + nn] = False
        fdofs = np.where(act)[0]
        Pf = Phi[fdofs].tocsc(); nzc = np.where(np.asarray(abs(Pf).sum(0)).ravel() > 0)[0]
        Pf = Pf[:, nzc].tocsr(); PfT = Pf.T.tocsr(); ncd = Pf.shape[1]
        # ---------- 积分聚合点 ----------
        T.tic('agg')
        if agg:
            h_ = grid['h']; ox_, oy_ = grid['origin']
            sx_ = np.floor((x[:, 0] - ox_) / h_ + 0.5).astype(np.int64) // agg
            sy_ = np.floor((x[:, 1] - oy_) / h_ + 0.5).astype(np.int64) // agg
            _, qi = np.unique((sx_ * 100003 + sy_) * 64 + mat_id, return_inverse=True); nq = qi.max() + 1
            qV = np.bincount(qi, Vp0, nq)
            xq = np.stack([np.bincount(qi, Vp0 * x[:, 0], nq), np.bincount(qi, Vp0 * x[:, 1], nq)], 1) / qV[:, None]
            qF0 = [np.bincount(qi, Vp0 * f, nq) / qV for f in F0]
            ql = np.bincount(qi, Vp0 * lam, nq) / qV; qm = np.bincount(qi, Vp0 * mu, nq) / qV
            qidx, _, qdx, qdy = shape_2d(xq, grid, 'qbspline')
        else:
            qidx, qdx, qdy, qF0, qV, ql, qm = idx, dNx, dNy, F0, Vp0, lam, mu
        qfi = qidx.ravel()
        T.tic('pattern')
        free_map = -np.ones(2 * nn, np.int64); free_map[fdofs] = np.arange(len(fdofs))
        pat = _pattern(qidx, nn, free_map)
        fext = PfT @ np.concatenate([mi * gx * fac, mi * gy * fac])[fdofs]
        U = np.zeros(ncd)
        for it in range(maxit + 1):
            T.tic('const')
            du = np.zeros(2 * nn); du[fdofs] = Pf @ U
            ux = du[:nn][qidx]; uy = du[nn:][qidx]
            a11 = 1 + np.sum(ux * qdx, 1); a12 = np.sum(ux * qdy, 1); a21 = np.sum(uy * qdx, 1); a22 = 1 + np.sum(uy * qdy, 1)
            Fq = [a11 * qF0[0] + a12 * qF0[2], a11 * qF0[1] + a12 * qF0[3], a21 * qF0[0] + a22 * qF0[2], a21 * qF0[1] + a22 * qF0[3]]
            s11, s12, s22, J = neo_cauchy(*Fq, ql, qm)
            Jr = a11 * a22 - a12 * a21
            if not np.isfinite(J).all() or J.min() <= 0:
                T.stop(); raise FloatingPointError(f'载荷步 {ls}: J<=0')
            T.tic('resid')
            P11 = s11 * a22 - s12 * a12; P12 = -s11 * a21 + s12 * a11; P21 = s12 * a22 - s22 * a12; P22 = -s12 * a21 + s22 * a11
            fx = np.bincount(qfi, ((qV * P11)[:, None] * qdx + (qV * P12)[:, None] * qdy).ravel(), nn)
            fy = np.bincount(qfi, ((qV * P21)[:, None] * qdx + (qV * P22)[:, None] * qdy).ravel(), nn)
            R = fext - PfT @ np.concatenate([fx, fy])[fdofs]
            rn = np.linalg.norm(R) / max(np.linalg.norm(fext), 1e-30)
            if verbose:
                print(f'  载荷步 {ls} 迭代 {it} |R|/|f|={rn:.2e}')
            if rn < tol:
                break
            if it == maxit:
                T.stop(); raise RuntimeError(f'载荷步 {ls} 约化 Newton 未收敛 {rn:.2e}')
            T.tic('tangent')
            dX = (a22[:, None] * qdx - a21[:, None] * qdy) / Jr[:, None]; dY = (-a12[:, None] * qdx + a11[:, None] * qdy) / Jr[:, None]
            vals = _tangent_vals(dX, dY, Jr * qV, ql / J, (qm - ql * np.log(J)) / J, s11, s12, s22)
            K = sp.csr_matrix((np.bincount(pat['inv'], vals[pat['keep']], pat['nnz']), pat['indices'], pat['indptr']),
                              shape=(pat['nf'], pat['nf']))
            T.tic('project')
            Kc = (PfT @ K @ Pf).tocsc()
            T.tic('solve')
            U = U + splu(Kc, permc_spec='MMD_AT_PLUS_A').solve(R)
        T.tic('update')
        du = np.zeros(2 * nn); du[fdofs] = Pf @ U
        ux = du[:nn][idx]; uy = du[nn:][idx]
        b11 = 1 + np.sum(ux * dNx, 1); b12 = np.sum(ux * dNy, 1); b21 = np.sum(uy * dNx, 1); b22 = 1 + np.sum(uy * dNy, 1)
        F = [b11 * F0[0] + b12 * F0[2], b11 * F0[1] + b12 * F0[3], b21 * F0[0] + b22 * F0[2], b21 * F0[1] + b22 * F0[3]]
        x[:, 0] += np.sum(N * ux, 1); x[:, 1] += np.sum(N * uy, 1)
        hist['newton'].append(it); hist['ndof'].append(ncd); hist['nq'].append(len(qV)); hist['tip'].append(x[par['monitor']].ravel().copy())
    T.stop()
    s11, s12, s22, J = neo_cauchy(*F, lam, mu)
    return dict(x=x, F=np.stack(F, 1), s=np.stack([s11, s12, s22], 1), hist={k: np.array(v) for k, v in hist.items()}, timer=T, Np=Np)
