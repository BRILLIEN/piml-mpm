# -*- coding: utf-8 -*-
"""非均质材料分布与粒子生成（显式/隐式共用）。"""
import numpy as np


def random_inclusions(xmin, xmax, ymin, ymax, frac, rmin, rmax, seed, gap=0.0):
    """在矩形内随机布置不重叠圆形夹杂，直至体积分数达到 frac。返回 (nc,3) [xc,yc,r]。"""
    rng = np.random.default_rng(seed)
    A = (xmax - xmin) * (ymax - ymin)
    C = []; area = 0.0; tries = 0
    while area < frac * A and tries < 200000:
        tries += 1
        r = rng.uniform(rmin, rmax)
        xc = rng.uniform(xmin + r, xmax - r); yc = rng.uniform(ymin + r, ymax - r)
        if C:
            c = np.array(C)
            if np.any(np.hypot(c[:, 0] - xc, c[:, 1] - yc) < c[:, 2] + r + gap):
                continue
        C.append((xc, yc, r)); area += np.pi * r * r
    return np.array(C).reshape(-1, 3)


def phase_of(X, incl):
    """粒子参考坐标 X 所在相：0 = 基体，1 = 夹杂。"""
    ph = np.zeros(len(X), dtype=np.int8)
    for xc, yc, r in incl:
        ph[(X[:, 0] - xc) ** 2 + (X[:, 1] - yc) ** 2 < r * r] = 1
    return ph


def particles_rect(xmin, xmax, ymin, ymax, h, ppc):
    """矩形区域内每个单元 ppc x ppc 规则布置粒子。返回坐标与每粒子体积。"""
    nx = int(round((xmax - xmin) / h)); ny = int(round((ymax - ymin) / h))
    d = h / ppc
    xs = xmin + d * (np.arange(nx * ppc) + 0.5)
    ys = ymin + d * (np.arange(ny * ppc) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], 1), d * d
