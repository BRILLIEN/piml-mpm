# -*- coding: utf-8 -*-
"""
显式多尺度 MPM（非学习加速方案 E3a）：
  粗尺度运动学 v_h = Φ V（Φ 为由粗单元内粒子材料分布局部凝聚得到的多尺度基），
  在"更新区间"内冻结粒子—网格映射（区间内为以区间起点构形为参考的完全拉格朗日格式），
  区间末更新粒子位置/速度/变形梯度，并按新的材料分布重建 Φ（状态相关局部算子）。

力学一致性
  * 质量：区间内细尺度集中质量 m_i 冻结（拉格朗日质量守恒）；粗尺度一致质量 M = Φ^T diag(m) Φ。
  * 惯性：粗尺度方程 M A = Φ^T (f_int + f_ext) 由细尺度 Galerkin 投影得到（Rayleigh–Ritz），
    不使用静力凝聚的"等效刚度"替代动力学；高频细尺度模态被滤除，故临界时间步增大。
  * 内力：区间内 f_i = −Σ_p V_p(t0) P_rel(F_p) ∇N_i(x_p(t0))，P_rel = J_rel σ F_rel^{-T}（相对 t0 的第一 PK 应力），
    与细尺度 USL 基准使用同一本构与同一粒子积分。
  * 运动学：Φ 满足单位分解、刚体运动再现；粒子层面插值仍为 C1 二次 B 样条。

每步计算（计时）：
  step_gather : 细节点位移 u_h = Φ U 并聚集到粒子求 D = Σ u_i ⊗ ∇N_i
  step_const  : F = (I + D) F0、Neo-Hookean 应力、P_rel
  step_scatter: 内力向细节点散射（2 次 bincount）
  step_coarse : 投影 Φ^T f、粗尺度速度/位移更新
区间更新（计时）：int_shape, int_basis, int_mass, int_update
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from common.shape import shape_2d
from common.material import neo_cauchy, neo_energy
from common.msbasis import build_basis, extend_small_cut
from common.timer import Timer


def run_explicit_ms(P, grid, par, n=4, K_int=50, quad='image', basis='ms', monitor=None, verbose=True,
                    Phi_provider=None):
    x = P['x'].copy(); v = P['v'].copy()
    m = P['m']; V0 = P['V0']; lam = P['lam']; mu = P['mu']
    Np = len(m); nn = grid['nn']
    F = [np.ones(Np), np.zeros(Np), np.zeros(Np), np.ones(Np)]
    gx, gy = par['gravity']; dt = par['dt']; nstep = par['nstep']; alpha = par['flip']
    out_every = par.get('out_every', 50)
    T = Timer()
    hist = dict(t=[], Ek=[], Es=[], Wg=[], mon=[], n_int=0)
    n = int(n)
    step = 0
    _, mat_id = np.unique(np.stack([lam, mu], 1), axis=0, return_inverse=True); mat_id = mat_id.ravel()
    while True:
        # ================= 区间更新 =================
        T.tic('int_shape')
        idx, N, dNx, dNy = shape_2d(x, grid, 'qbspline'); fi = idx.ravel()
        T.tic('int_basis')
        Vcur = V0 * (F[0] * F[3] - F[1] * F[2])
        if Phi_provider is not None:              # 学习型（PIML）基或其他外部提供者
            B = Phi_provider(x, Vcur, lam, mu, grid, n)
        elif basis == 'ms':
            B = build_basis(x, Vcur, lam, mu, grid, n, quad=quad, fixed_nodes=par['fixed_nodes'])
        elif basis == 'os':                        # 边自适应（两单元过采样）多尺度基
            from common.msbasis_os import build_basis_os
            B = build_basis_os(x, Vcur, lam, mu, grid, n, fixed_nodes=par['fixed_nodes'])
        elif basis == 'os_piml':                   # 同上，但边条带由 PIML 网络预测（含固定节点的边回退精确）
            from common.msbasis_os import build_basis_os
            from ml.train_edge import PIMLEdge
            if not hasattr(run_explicit_ms, '_edge_models'):
                run_explicit_ms._edge_models = {e: PIMLEdge(f'ml/models/edge_n{n}_{e}{par.get("edge_suffix", "")}.npz') for e in ('x', 'y')}
            B = build_basis_os(x, Vcur, lam, mu, grid, n, fixed_nodes=par['fixed_nodes'],
                               edge_models=run_explicit_ms._edge_models)
        else:                                      # 对照：纯粗尺度双线性（不含非均质信息）
            from common.msbasis import coarse_grid, coarse_global
            cg = coarse_grid(grid, n); Pb = coarse_global(grid, cg, np.arange(nn), 'bilinear')
            Pb = Pb.tolil(); Pb[par['fixed_nodes'], :] = 0; Pb = Pb.tocsr()      # 细尺度固定节点行置零
            B = dict(Phi=sp.block_diag([Pb, Pb]).tocsr(), cg=cg)
        T.tic('int_mass')
        mN = N * m[:, None]
        mi = np.bincount(fi, mN.ravel(), nn)
        Phi = B['Phi']; cg = B['cg']; nc = cg['nc']
        if par.get('extend', True):            # 小切割粗自由度外推约束
            T.tic('int_extend')
            Phi, einfo = extend_small_cut(Phi, cg, np.concatenate([mi, mi]), theta=par.get('theta', 0.3))
            T.tic('int_mass')
        PhiT = Phi.T.tocsr()
        pxi = np.bincount(fi, (mN * v[:, 0:1]).ravel(), nn); pyi = np.bincount(fi, (mN * v[:, 1:2]).ravel(), nn)
        mh = np.concatenate([mi, mi])
        # 粗尺度一致质量 Mc = Φ^T M_h Φ（Galerkin）。多尺度基含负值，行和集中质量可能近零/失去能量有界性，
        # 而粗系统规模很小（~细节点数/n²），一次分解、每步回代的代价远小于粒子运算。
        Mcons = (PhiT @ sp.diags(mh) @ Phi).tocsc()
        dM = Mcons.diagonal()
        fr = np.where(dM > 1e-10 * dM.max())[0]              # 有效粗自由度（细尺度位移边界已由 Φ 精确施加）
        LU = splu(Mcons[fr][:, fr].tocsc())
        def msolve(r):
            out = np.zeros(2 * nc); out[fr] = LU.solve(r[fr]); return out
        Vc = msolve(PhiT @ np.concatenate([pxi, pyi]))        # 粗尺度速度：质量加权 L2 投影
        Vstart = Vc.copy()
        fext = PhiT @ np.concatenate([mi * gx, mi * gy])
        Uc = np.zeros(2 * nc)
        F0 = [f.copy() for f in F]; Vp0 = Vcur
        # ---- 粒子积分聚合（超降阶，免训练）：同一细区间块(agg×agg 个区间)内同一材料相的粒子合并为一个积分点 ----
        agg = par.get('agg', 0)
        if agg:
            T.tic('int_agg')
            h_ = grid['h']; ox_, oy_ = grid['origin']
            sx_ = np.floor((x[:, 0] - ox_) / h_ + 0.5).astype(np.int64) // agg
            sy_ = np.floor((x[:, 1] - oy_) / h_ + 0.5).astype(np.int64) // agg
            key = (sx_ * 100003 + sy_) * 64 + mat_id
            _, qi = np.unique(key, return_inverse=True); nq = qi.max() + 1
            qV = np.bincount(qi, Vp0, nq)
            xq = np.stack([np.bincount(qi, Vp0 * x[:, 0], nq), np.bincount(qi, Vp0 * x[:, 1], nq)], 1) / qV[:, None]
            qF0 = [np.bincount(qi, Vp0 * f, nq) / qV for f in F0]
            qlam = np.bincount(qi, Vp0 * lam, nq) / qV; qmu = np.bincount(qi, Vp0 * mu, nq) / qV
            qidx, _, qdNx, qdNy = shape_2d(xq, grid, 'qbspline'); qfi = qidx.ravel()
        else:
            qidx, qdNx, qdNy, qfi, qF0, qV, qlam, qmu = idx, dNx, dNy, fi, F0, Vp0, lam, mu
        hist.setdefault('nq', []).append(len(qV))

        def full_F(Ucur):
            uh_ = Phi @ Ucur; ux_ = uh_[:nn][idx]; uy_ = uh_[nn:][idx]
            b11 = 1 + np.sum(ux_ * dNx, 1); b12 = np.sum(ux_ * dNy, 1); b21 = np.sum(uy_ * dNx, 1); b22 = 1 + np.sum(uy_ * dNy, 1)
            return [b11 * F0[0] + b12 * F0[2], b11 * F0[1] + b12 * F0[3], b21 * F0[0] + b22 * F0[2], b21 * F0[1] + b22 * F0[3]]
        x0 = x.copy()
        nsub = min(K_int, nstep - step)
        hist['n_int'] += 1
        # ================= 区间内时间推进（完全拉格朗日，参考构形 t0） =================
        for k in range(nsub + 1):
            if (step % out_every == 0 and k < nsub) or step == nstep:     # 区间末步由下一区间首步记录，避免重复
                T.tic('output')
                uh = Phi @ Uc; vh = Phi @ Vc
                ux = np.sum(N * uh[:nn][idx], 1); uy = np.sum(N * uh[nn:][idx], 1)
                vx = np.sum(N * vh[:nn][idx], 1); vy = np.sum(N * vh[nn:][idx], 1)
                Ek = 0.5 * np.sum(m * (vx ** 2 + vy ** 2))
                Es = np.sum(V0 * neo_energy(*full_F(Uc), lam, mu))
                Wg = -np.sum(m * (gx * (x0[:, 0] + ux - P['x'][:, 0]) + gy * (x0[:, 1] + uy - P['x'][:, 1])))
                hist['t'].append(step * dt); hist['Ek'].append(Ek); hist['Es'].append(Es); hist['Wg'].append(Wg)
                if monitor is not None:
                    # 与细尺度基准相同的列格式：[x,y]×监测点, [vx,vy]×监测点, s11,s12,s22（此处应力不记录，填 0）
                    xm = np.stack([x0[monitor, 0] + ux[monitor], x0[monitor, 1] + uy[monitor]], 1).ravel()
                    vm = np.stack([vx[monitor], vy[monitor]], 1).ravel()
                    hist['mon'].append(np.concatenate([xm, vm, np.zeros(3 * len(monitor))]))
                if verbose and step % (out_every * 20) == 0:
                    print(f'  step {step:7d}  t={step*dt:.4e}  Ek={Ek:.4e}  Es={Es:.4e}  Etot={Ek+Es+Wg:.4e}', flush=True)
            if k == nsub:
                break
            # ---- 细节点位移 -> 粒子位移梯度 ----
            T.tic('step_gather')
            uh = Phi @ Uc
            uxp = uh[:nn][qidx]; uyp = uh[nn:][qidx]
            D11 = np.sum(uxp * qdNx, 1); D12 = np.sum(uxp * qdNy, 1)
            D21 = np.sum(uyp * qdNx, 1); D22 = np.sum(uyp * qdNy, 1)
            # ---- 本构 ----
            T.tic('step_const')
            a11 = 1 + D11; a12 = D12; a21 = D21; a22 = 1 + D22                # F_rel
            Fq = [a11 * qF0[0] + a12 * qF0[2], a11 * qF0[1] + a12 * qF0[3],
                  a21 * qF0[0] + a22 * qF0[2], a21 * qF0[1] + a22 * qF0[3]]
            s11, s12, s22, J = neo_cauchy(*Fq, qlam, qmu)
            Jr = a11 * a22 - a12 * a21
            # P_rel = Jr σ F_rel^{-T}；F_rel^{-T} = [[a22,-a21],[-a12,a11]]/Jr
            P11 = s11 * a22 - s12 * a12; P12 = -s11 * a21 + s12 * a11
            P21 = s12 * a22 - s22 * a12; P22 = -s12 * a21 + s22 * a11
            W = qV
            # ---- 内力散射 ----
            T.tic('step_scatter')
            fx = np.bincount(qfi, (-(W * P11)[:, None] * qdNx - (W * P12)[:, None] * qdNy).ravel(), nn)
            fy = np.bincount(qfi, (-(W * P21)[:, None] * qdNx - (W * P22)[:, None] * qdNy).ravel(), nn)
            # ---- 粗尺度更新（辛欧拉 / 中心差分） ----
            T.tic('step_coarse')
            Fc = PhiT @ np.concatenate([fx, fy]) + fext
            Vc = Vc + dt * msolve(Fc)
            Uc = Uc + dt * Vc
            step += 1
            if not np.isfinite(J).all() or J.min() <= 0:
                T.stop(); raise FloatingPointError(f'J<=0 或非有限，step {step}')
        # ================= 区间末：粒子更新 =================
        T.tic('int_update')
        F = full_F(Uc)                                   # 全体粒子变形梯度（每区间一次）
        uh = Phi @ Uc; vh = Phi @ Vc; dvh = Phi @ (Vc - Vstart)
        x[:, 0] = x0[:, 0] + np.sum(N * uh[:nn][idx], 1); x[:, 1] = x0[:, 1] + np.sum(N * uh[nn:][idx], 1)
        vpicx = np.sum(N * vh[:nn][idx], 1); vpicy = np.sum(N * vh[nn:][idx], 1)
        v[:, 0] = alpha * (v[:, 0] + np.sum(N * dvh[:nn][idx], 1)) + (1 - alpha) * vpicx
        v[:, 1] = alpha * (v[:, 1] + np.sum(N * dvh[nn:][idx], 1)) + (1 - alpha) * vpicy
        if step >= nstep:
            break
    T.stop()
    s11, s12, s22, J = neo_cauchy(*F, lam, mu)
    for k in hist:
        if k != 'n_int':
            hist[k] = np.array(hist[k])
    return dict(x=x, v=v, F=np.stack(F, 1), s=np.stack([s11, s12, s22], 1), hist=hist, timer=T, Np=Np,
                nc=nc, nstep=nstep)
