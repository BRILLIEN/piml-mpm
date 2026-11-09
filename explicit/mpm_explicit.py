# -*- coding: utf-8 -*-
"""
显式 MPM 基准求解器（USL 格式，FLIP/PIC 混合，Neo-Hookean）。

每个时间步的计算环节（分别计时）：
  1 shape   : 形函数及梯度求值（粒子 -> 支撑节点）
  2 p2g     : 质量、动量、内力、外力向网格传递（accumarray/bincount）
  3 grid    : 网格动量更新与边界条件
  4 g2p     : 网格向粒子传递速度/加速度、位置更新、速度梯度
  5 const   : 变形梯度与本构更新
  6 output  : 能量/监测量记录
所有运算为向量化数组运算；P2G 散射使用 np.bincount（对应 MATLAB accumarray）。
"""
import numpy as np
from common.shape import shape_2d
from common.material import neo_cauchy, neo_energy
from common.timer import Timer


def run_explicit(P, grid, par, kind='qbspline', monitor=None, verbose=True):
    """P: 粒子数据 dict(x, v, m, V0, lam, mu)；grid：背景网格；par：求解参数。
    返回结果 dict（时程、能量、计时等）。"""
    x = P['x'].copy(); v = P['v'].copy()
    m = P['m']; V0 = P['V0']; lam = P['lam']; mu = P['mu']
    Np = len(m); nn = grid['nn']
    F11 = np.ones(Np); F12 = np.zeros(Np); F21 = np.zeros(Np); F22 = np.ones(Np)
    s11 = np.zeros(Np); s12 = np.zeros(Np); s22 = np.zeros(Np)
    V = V0.copy()
    gx, gy = par['gravity']
    dt = par['dt']; nstep = par['nstep']; alpha = par['flip']
    fixed = par['fixed_nodes']                   # 固定节点（两个方向）
    mtol = 1e-12 * m.max()
    out_every = par.get('out_every', 50)
    T = Timer()
    hist = dict(t=[], Ek=[], Es=[], Wg=[], mon=[])
    for n in range(nstep + 1):
        # ---- 记录（Wg 为重力势能，总能量 Etot = Ek + Es + Wg）----
        if n % out_every == 0 or n == nstep:
            T.tic('output')
            Ek = 0.5 * np.sum(m * (v[:, 0] ** 2 + v[:, 1] ** 2))
            Es = np.sum(V0 * neo_energy(F11, F12, F21, F22, lam, mu))
            Wg = -np.sum(m * (gx * (x[:, 0] - P['x'][:, 0]) + gy * (x[:, 1] - P['x'][:, 1])))
            hist['t'].append(n * dt); hist['Ek'].append(Ek); hist['Es'].append(Es); hist['Wg'].append(Wg)
            if monitor is not None:
                hist['mon'].append(np.concatenate([x[monitor].ravel(), v[monitor].ravel(),
                                                   s11[monitor], s12[monitor], s22[monitor]]))
            if verbose and n % (out_every * 20) == 0:
                print(f'  step {n:7d}  t={n*dt:.4e}  Ek={Ek:.4e}  Es={Es:.4e}  Etot={Ek+Es+Wg:.4e}', flush=True)
        if n == nstep:
            break
        # ---- 1 形函数 ----
        T.tic('shape')
        idx, N, dNx, dNy = shape_2d(x, grid, kind)
        fi = idx.ravel()
        # ---- 2 P2G ----
        T.tic('p2g')
        mN = N * m[:, None]
        mI = np.bincount(fi, mN.ravel(), nn)
        pxI = np.bincount(fi, (mN * v[:, 0:1]).ravel(), nn)
        pyI = np.bincount(fi, (mN * v[:, 1:2]).ravel(), nn)
        Vs11 = (V * s11)[:, None]; Vs12 = (V * s12)[:, None]; Vs22 = (V * s22)[:, None]
        fxI = np.bincount(fi, (-(Vs11 * dNx + Vs12 * dNy) + mN * gx).ravel(), nn)
        fyI = np.bincount(fi, (-(Vs12 * dNx + Vs22 * dNy) + mN * gy).ravel(), nn)
        # ---- 3 网格更新 ----
        T.tic('grid')
        act = mI > mtol
        inv = np.zeros(nn); inv[act] = 1.0 / mI[act]
        vxo = pxI * inv; vyo = pyI * inv
        vxn = vxo + dt * fxI * inv; vyn = vyo + dt * fyI * inv
        vxn[fixed] = 0.0; vyn[fixed] = 0.0
        axI = (vxn - vxo) / dt; ayI = (vyn - vyo) / dt
        # ---- 4 G2P ----
        T.tic('g2p')
        vxn_p = vxn[idx]; vyn_p = vyn[idx]
        vpicx = np.sum(N * vxn_p, 1); vpicy = np.sum(N * vyn_p, 1)
        apx = np.sum(N * axI[idx], 1); apy = np.sum(N * ayI[idx], 1)
        v[:, 0] = alpha * (v[:, 0] + dt * apx) + (1 - alpha) * vpicx
        v[:, 1] = alpha * (v[:, 1] + dt * apy) + (1 - alpha) * vpicy
        x[:, 0] += dt * vpicx; x[:, 1] += dt * vpicy
        if par.get('scheme', 'USL') == 'MUSL':
            # MUSL：用更新后的粒子动量重新映射网格速度，再求速度梯度（抑制小质量节点引起的速度放大）
            T.tic('musl')
            vxn = np.bincount(fi, (mN * v[:, 0:1]).ravel(), nn) * inv
            vyn = np.bincount(fi, (mN * v[:, 1:2]).ravel(), nn) * inv
            vxn[fixed] = 0.0; vyn[fixed] = 0.0
            vxn_p = vxn[idx]; vyn_p = vyn[idx]
            T.tic('g2p')
        L11 = np.sum(vxn_p * dNx, 1); L12 = np.sum(vxn_p * dNy, 1)
        L21 = np.sum(vyn_p * dNx, 1); L22 = np.sum(vyn_p * dNy, 1)
        # ---- 5 本构 ----
        T.tic('const')
        a11 = 1 + dt * L11; a12 = dt * L12; a21 = dt * L21; a22 = 1 + dt * L22
        F11, F12, F21, F22 = (a11 * F11 + a12 * F21, a11 * F12 + a12 * F22,
                              a21 * F11 + a22 * F21, a21 * F12 + a22 * F22)
        s11, s12, s22, J = neo_cauchy(F11, F12, F21, F22, lam, mu)
        V = J * V0
        if not np.isfinite(J).all() or J.min() <= 0:
            T.stop()
            raise FloatingPointError(f'J<=0 或非有限，step {n}')
    T.stop()
    for k in hist:
        hist[k] = np.array(hist[k])
    return dict(x=x, v=v, F=np.stack([F11, F12, F21, F22], 1), s=np.stack([s11, s12, s22], 1),
                hist=hist, timer=T, Np=Np, nn_active=int(act.sum()), nstep=nstep)
