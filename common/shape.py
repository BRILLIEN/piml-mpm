# -*- coding: utf-8 -*-
"""
公共基础函数：背景网格形函数及其梯度（二维张量积）。

说明
----
* 网格为均匀结构网格，节点 (i, j) 坐标 = origin + (i*h, j*h)，全局编号 I = i + j*nx。
* 'linear'  : 双线性（C0，梯度在网格线处跳变 -> 经典 cell-crossing 来源）
* 'qbspline': 均匀二次 B 样条（C1，梯度连续），每个粒子 3x3=9 个支撑节点
* 所有运算均为向量化数组运算，在 MATLAB 中可用相同形式实现
  （floor / 逐元素乘法 / 索引），不依赖 Python 独有的加速框架。

返回
----
idx : (Np, nn) int   支撑节点全局编号
N   : (Np, nn)       形函数值
dNx : (Np, nn)       x 方向导数
dNy : (Np, nn)       y 方向导数
"""
import numpy as np


def _basis_1d(xi, kind):
    """一维基函数。xi 为以网格间距无量纲化的坐标 (Np,)。
    返回 base (Np,) 起始节点号，w (Np,k)，dw (Np,k)（对 xi 的导数）。"""
    if kind == 'linear':
        base = np.floor(xi).astype(np.int64)
        f = xi - base
        w = np.stack([1.0 - f, f], axis=1)
        dw = np.stack([-np.ones_like(f), np.ones_like(f)], axis=1)
    elif kind == 'qbspline':
        base = np.floor(xi - 0.5).astype(np.int64)
        f = xi - base                      # f ∈ [0.5, 1.5)
        w = np.stack([0.5 * (1.5 - f) ** 2,
                      0.75 - (f - 1.0) ** 2,
                      0.5 * (f - 0.5) ** 2], axis=1)
        dw = np.stack([f - 1.5,
                       -2.0 * (f - 1.0),
                       f - 0.5], axis=1)
    else:
        raise ValueError(kind)
    return base, w, dw


def shape_2d(xp, grid, kind='qbspline'):
    """二维张量积形函数。xp: (Np,2) 粒子坐标；grid: dict(h, origin, nx, ny)。"""
    h = grid['h']
    ox, oy = grid['origin']
    nx = grid['nx']
    bx, wx, dwx = _basis_1d((xp[:, 0] - ox) / h, kind)
    by, wy, dwy = _basis_1d((xp[:, 1] - oy) / h, kind)
    k = wx.shape[1]
    ox_ = np.arange(k)
    # (Np, k, k)：第一个 k 为 y 偏移，第二个为 x 偏移，展平后 nn = k*k
    ii = bx[:, None, None] + ox_[None, None, :]
    jj = by[:, None, None] + ox_[None, :, None]
    idx = (ii + jj * nx).reshape(len(xp), -1)
    N = (wy[:, :, None] * wx[:, None, :]).reshape(len(xp), -1)
    dNx = (wy[:, :, None] * dwx[:, None, :]).reshape(len(xp), -1) / h
    dNy = (dwy[:, :, None] * wx[:, None, :]).reshape(len(xp), -1) / h
    return idx, N, dNx, dNy


def make_grid(xmin, xmax, ymin, ymax, h, pad=3):
    """生成覆盖 [xmin,xmax]x[ymin,ymax] 的背景网格（四周各留 pad 层单元）。"""
    ox = xmin - pad * h
    oy = ymin - pad * h
    nx = int(np.ceil((xmax - xmin) / h)) + 2 * pad + 1
    ny = int(np.ceil((ymax - ymin) / h)) + 2 * pad + 1
    xs = ox + h * np.arange(nx)
    ys = oy + h * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)          # 行 = j，列 = i，展平后 I = i + j*nx
    return dict(h=h, origin=(ox, oy), nx=nx, ny=ny, nn=nx * ny,
                xn=np.stack([X.ravel(), Y.ravel()], axis=1))
