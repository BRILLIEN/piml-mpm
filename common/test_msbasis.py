# -*- coding: utf-8 -*-
"""多尺度基检验：单位分解、刚体转动、均质时一次场再现、非均质时一次场偏差（预期非零）。
运行：python3 -m common.test_msbasis"""
import time
import numpy as np
from common.shape import make_grid, shape_2d
from common.material import lame
from common.hetero import random_inclusions, phase_of, particles_rect
from common.msbasis import build_basis


def fields(B, grid, xp, active, deep):
    Phi = B['Phi']; xc = B['cg']['xc']; nc = B['cg']['nc']; nn = grid['nn']; xn = grid['xn']
    def interp(Vx, Vy):
        u = Phi @ np.concatenate([Vx, Vy]); return u[:nn], u[nn:]
    e = {}
    ux, uy = interp(np.ones(nc), np.zeros(nc))
    e['PU'] = max(np.abs(ux[active] - 1).max(), np.abs(uy[active]).max())
    ux, uy = interp(-xc[:, 1], xc[:, 0])
    e['rot'] = max(np.abs(ux[active] + xn[active, 1]).max(), np.abs(uy[active] - xn[active, 0]).max())
    ux, uy = interp(0.3 * xc[:, 0] + 0.1 * xc[:, 1], -0.2 * xc[:, 0] + 0.5 * xc[:, 1])
    # 一次场：自由表面附近线性场不满足无牵引条件，凝聚解本就不应是线性，故只在远离表面的节点检验
    e['lin'] = max(np.abs(ux[deep] - (0.3 * xn[deep, 0] + 0.1 * xn[deep, 1])).max(),
                   np.abs(uy[deep] - (-0.2 * xn[deep, 0] + 0.5 * xn[deep, 1])).max())
    return e


if __name__ == '__main__':
    h = 0.01; L, H = 1.0, 0.5
    grid = make_grid(-0.1, 1.5, -0.5, 0.8, h)
    X, Vp = particles_rect(0, L, 0, H, h, 2)
    # 施加随机扰动位置模拟粒子迁移后的状态
    rng = np.random.default_rng(3); X = X + rng.uniform(-0.3, 0.3, X.shape) * h * 0.5
    incl = random_inclusions(0, L, 0, H, 0.3, 0.012, 0.03, 1)
    ph = phase_of(X, incl)
    lm, mm = lame(1e6, 0.3); li, mi = lame(1e7, 0.3)
    Vv = np.full(len(X), Vp)
    idx, N, _, _ = shape_2d(X, grid)
    active = np.unique(idx)
    ok = True
    for n, quad in [(4, 'particle'), (4, 'gauss'), (8, 'gauss')]:
        for label, lam, mu in [('均质', np.full(len(X), lm), np.full(len(X), mm)),
                               ('非均质', np.where(ph == 1, li, lm), np.where(ph == 1, mi, mm))]:
            t0 = time.perf_counter()
            B = build_basis(X, Vv, lam, mu, grid, n, quad=quad)
            t1 = time.perf_counter() - t0
            xn = grid['xn']; Hc = n * h
            deep = active[(xn[active, 0] > 2 * Hc) & (xn[active, 0] < L - 2 * Hc) &
                          (xn[active, 1] > 2 * Hc) & (xn[active, 1] < H - 2 * Hc)]
            e = fields(B, grid, X, active, deep)
            print(f'n={n} {quad} {label}: ncell={B["ncell"]}, nnz(Φ)={B["Phi"].nnz}, 构造耗时 {t1:.3f}s,',
                  {k: f'{v:.2e}' for k, v in e.items()})
            ok &= e['PU'] < 1e-8 and e['rot'] < 1e-8
            if label == '均质' and quad == 'gauss':
                ok &= e['lin'] < 1e-8
            else:
                ok &= e['lin'] > 1e-6          # 非均质时一次场不应被再现
    print('PASS' if ok else 'FAIL')
