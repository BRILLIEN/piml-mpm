# -*- coding: utf-8 -*-
"""
Cell-crossing 机理检验（独立于动力学）：均匀应力 σ 的块体，其粒子格点整体平移 δ ∈ [0, h)。
精确解：内部节点内力 f_i = −∫ σ·∇N_i dV = 0。粒子积分得到 f_i = −Σ_p V_p σ·∇N_i(x_p)，
其非零部分即粒子跨格（及粒子积分）误差。统计远离表面的内部节点的 max|f_i| / (σ h)（无量纲）。
同时在形变后粒子分布（随机扰动 / 非均匀拉伸）下比较。
运行：python3 -m explicit.cellcross_test
"""
import json
import numpy as np
from common.shape import make_grid, shape_2d


def residual(kind, delta, ppc, h=1.0, nc=20, perturb=0.0, stretch=1.0, seed=0):
    g = make_grid(-3 * h, (nc + 3) * h, -3 * h, (nc + 3) * h, h, pad=3)
    d = h / ppc
    s = d * (np.arange(nc * ppc) + 0.5)
    X, Y = np.meshgrid(s, s); xp = np.stack([X.ravel(), Y.ravel()], 1)
    xp[:, 0] = xp[:, 0] * stretch                                  # 非均匀间距（拉伸）
    xp = xp + delta * h
    if perturb > 0:
        xp = xp + np.random.default_rng(seed).uniform(-perturb, perturb, xp.shape) * d
    Vp = np.full(len(xp), d * d * stretch)
    idx, N, dNx, dNy = shape_2d(xp, g, kind)
    s11, s12, s22 = 1.0, 0.3, -0.5                                   # 均匀应力
    fx = np.bincount(idx.ravel(), (-Vp[:, None] * (s11 * dNx + s12 * dNy)).ravel(), g['nn'])
    fy = np.bincount(idx.ravel(), (-Vp[:, None] * (s12 * dNx + s22 * dNy)).ravel(), g['nn'])
    xn = g['xn']; L = nc * h * min(stretch, 1.0)
    inner = (xn[:, 0] > 3 * h) & (xn[:, 0] < L - 3 * h) & (xn[:, 1] > 3 * h) & (xn[:, 1] < nc * h - 3 * h)
    return float(np.max(np.hypot(fx[inner], fy[inner])) / h)


if __name__ == '__main__':
    out = {}
    deltas = np.linspace(0, 1, 21)[:-1]
    print('均匀应力块体平移 δ（单位 h）时内部节点的最大虚假内力 max|f|/(σh)：')
    for kind in ['linear', 'qbspline']:
        for ppc in [2, 3, 4]:
            r = [residual(kind, dd, ppc) for dd in deltas]
            out[f'{kind}_ppc{ppc}_translate'] = r
            print(f'  {kind:9s} ppc={ppc}: 平移扫描 max={max(r):.2e}  min={min(r):.2e}')
        for ppc in [2, 4]:
            rp = [residual(kind, 0.13, ppc, perturb=0.25, seed=k) for k in range(5)]
            rs = residual(kind, 0.13, ppc, stretch=1.07)
            out[f'{kind}_ppc{ppc}_perturb'] = rp; out[f'{kind}_ppc{ppc}_stretch'] = rs
            print(f'  {kind:9s} ppc={ppc}: 随机扰动(±0.25d) max={max(rp):.2e}；7% 拉伸 {rs:.2e}')
    json.dump(dict(deltas=deltas.tolist(), **out), open('results/explicit/cellcross_test.json', 'w'), indent=1)
