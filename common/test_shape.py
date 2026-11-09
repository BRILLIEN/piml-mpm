# -*- coding: utf-8 -*-
"""形函数检验：单位分解、一次场再现、梯度和为零、梯度一次再现、跨格连续性。
运行：python3 -m common.test_shape"""
import numpy as np
from common.shape import shape_2d, make_grid


def check(kind, rng):
    g = make_grid(0, 1, 0, 1, 0.1)
    xp = rng.uniform(0.0, 1.0, size=(20000, 2))
    idx, N, dNx, dNy = shape_2d(xp, g, kind)
    xI = g['xn'][idx]                                 # (Np,nn,2)
    e = dict(
        pou=np.abs(N.sum(1) - 1).max(),
        lin_x=np.abs((N * xI[..., 0]).sum(1) - xp[:, 0]).max(),
        lin_y=np.abs((N * xI[..., 1]).sum(1) - xp[:, 1]).max(),
        grad_sum=max(np.abs(dNx.sum(1)).max(), np.abs(dNy.sum(1)).max()),
        grad_lin=max(np.abs((dNx * xI[..., 0]).sum(1) - 1).max(),
                     np.abs((dNy * xI[..., 1]).sum(1) - 1).max(),
                     np.abs((dNx * xI[..., 1]).sum(1)).max()))
    # 跨格连续性：粒子沿 x 穿越网格线 x=0.5，比较两侧极限的 N 与 dN（对同一节点）
    eps = 1e-9
    jumpN, jumpG = 0.0, 0.0
    for xl in [0.5, 0.55]:   # 0.5 为线性单元界面；0.55 为二次 B 样条节点间中点（其"单元"界面）
        xa = np.array([[xl - eps, 0.333]]); xb = np.array([[xl + eps, 0.333]])
        ia, Na, gxa, _ = shape_2d(xa, g, kind)
        ib, Nb, gxb, _ = shape_2d(xb, g, kind)
        va = np.zeros(g['nn']); vb = np.zeros(g['nn'])
        np.add.at(va, ia[0], Na[0]); np.add.at(vb, ib[0], Nb[0])
        jumpN = max(jumpN, np.abs(va - vb).max())
        va[:] = 0; vb[:] = 0
        np.add.at(va, ia[0], gxa[0]); np.add.at(vb, ib[0], gxb[0])
        jumpG = max(jumpG, np.abs(va - vb).max() * g['h'])   # 无量纲化（×h）
    e['jump_N'] = jumpN
    e['jump_dN*h'] = jumpG
    return e


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    ok = True
    for kind in ['linear', 'qbspline']:
        e = check(kind, rng)
        print(kind, {k: f'{v:.2e}' for k, v in e.items()})
        ok &= all(e[k] < 1e-10 for k in ['pou', 'lin_x', 'lin_y', 'grad_sum', 'grad_lin'])
        ok &= e['jump_N'] < 1e-6        # ±1e-9 偏移 × 梯度 O(1/h) 带来的 O(1e-8) 残差
        if kind == 'qbspline':
            ok &= e['jump_dN*h'] < 1e-6
        else:
            ok &= e['jump_dN*h'] > 0.5      # 线性形函数梯度必然跳变（预期）
    print('PASS' if ok else 'FAIL')
