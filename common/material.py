# -*- coding: utf-8 -*-
"""可压缩 Neo-Hookean 材料（二维平面应变），显式与隐式路线共用。
ψ(F) = μ/2 (tr C − 2) − μ ln J + λ/2 (ln J)^2
σ   = μ/J (b − I) + λ ln J / J · I
所有函数对粒子数组向量化：F 为 (Np,2,2) 或分量形式。"""
import numpy as np


def lame(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def neo_cauchy(F11, F12, F21, F22, lam, mu):
    """返回 Cauchy 应力分量 (s11, s12, s22) 与 J。分量形式避免小矩阵循环。"""
    J = F11 * F22 - F12 * F21
    b11 = F11 * F11 + F12 * F12
    b12 = F11 * F21 + F12 * F22
    b22 = F21 * F21 + F22 * F22
    lnJ = np.log(J)
    a = mu / J
    c = lam * lnJ / J
    return a * (b11 - 1) + c, a * b12, a * (b22 - 1) + c, J


def neo_energy(F11, F12, F21, F22, lam, mu):
    J = F11 * F22 - F12 * F21
    I1 = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2
    lnJ = np.log(J)
    return 0.5 * mu * (I1 - 2) - mu * lnJ + 0.5 * lam * lnJ ** 2
