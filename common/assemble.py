# -*- coding: utf-8 -*-
"""全局稀疏组装（细尺度）：线弹性刚度、集中质量；以及广义特征值上界估计（临界时间步）。
自由度顺序 [ux(全部节点), uy(全部节点)]。"""
import numpy as np
import scipy.sparse as sp
from common.shape import shape_2d


def elastic_K(xp, Vp, lam, mu, grid, chunk=100000):
    nn = grid['nn']; mats = []
    for s in range(0, len(xp), chunk):
        e = slice(s, s + chunk)
        idx, N, dNx, dNy = shape_2d(xp[e], grid, 'qbspline')
        V = Vp[e][:, None, None]; L = lam[e][:, None, None]; M = mu[e][:, None, None]
        gxi = dNx[:, :, None]; gyi = dNy[:, :, None]; gxj = dNx[:, None, :]; gyj = dNy[:, None, :]
        Kxx = V * ((L + 2 * M) * gxi * gxj + M * gyi * gyj)
        Kxy = V * (L * gxi * gyj + M * gyi * gxj)
        Kyx = V * (L * gyi * gxj + M * gxi * gyj)
        Kyy = V * ((L + 2 * M) * gyi * gyj + M * gxi * gxj)
        r = np.broadcast_to(idx[:, :, None], Kxx.shape).ravel(); c = np.broadcast_to(idx[:, None, :], Kxx.shape).ravel()
        rows = np.concatenate([r, r, r + nn, r + nn]); cols = np.concatenate([c, c + nn, c, c + nn])
        vals = np.concatenate([Kxx.ravel(), Kxy.ravel(), Kyx.ravel(), Kyy.ravel()])
        mats.append(sp.csr_matrix((vals, (rows, cols)), shape=(2 * nn, 2 * nn)))
    return sum(mats[1:], mats[0])


def lumped_mass(xp, m, grid):
    idx, N, _, _ = shape_2d(xp, grid, 'qbspline')
    mi = np.bincount(idx.ravel(), (N * m[:, None]).ravel(), grid['nn'])
    return np.concatenate([mi, mi])


def max_eig(K, M, free):
    """λ_max(M^{-1}K)（限于自由自由度），临界时间步 Δt_cr = 2/sqrt(λ_max)。
    M 可为对角（一维数组，集中质量）或稀疏矩阵（一致质量）。Lanczos（eigsh，对应 MATLAB eigs）。"""
    from scipy.sparse.linalg import eigsh
    Kf = K[free][:, free]
    if sp.issparse(M):
        Mf = M[free][:, free].tocsc()
        return float(eigsh(Kf, k=1, M=Mf, which='LA', return_eigenvectors=False, tol=1e-6)[0])
    s = 1.0 / np.sqrt(M[free])
    S = sp.diags(s) @ Kf @ sp.diags(s)
    return float(eigsh(S, k=1, which='LA', return_eigenvectors=False, tol=1e-6)[0])
