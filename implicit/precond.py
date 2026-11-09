# -*- coding: utf-8 -*-
"""
隐式路线预条件器（作用于自由自由度上的切线 K）。
  ilu      : 不完全 LU（spilu，对应 MATLAB ilu）
  twolevel : 两级对称乘性预条件  z = S r；z += P Kc⁻¹ Pᵀ (r − K z)；z += Sᵀ (r − K z)
             S 为 ILU 平滑，P 为粗空间（限制到自由自由度），Kc = Pᵀ K P（Galerkin 粗算子，精确分解）。
             粗空间：'ms'（多尺度数值形函数，含非均质信息）或 'bilinear'/'qbspline'（纯几何粗化，对照）。
设计意图：保持细尺度解精确（PCG 收敛到相同容差），通过粗空间捕获非均质低能模态减少迭代次数；
机制 = "减少迭代次数"，与"降自由度"（粗空间直接求解，近似解）严格区分。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu, splu, LinearOperator
from common.msbasis import build_basis, coarse_grid, coarse_global, extend_small_cut


ILU_OPTS = dict(permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
# 对称模式（无主元交换、对称排序）使 ILU 近似 LDLᵀ，预条件器近似对称（PCG 要求）。默认 SuperLU ILU 带主元，PCG 不收敛。


def ilu_builder(drop_tol=1e-3, fill_factor=2.0):
    def build(K, ctx):
        lu = spilu(K.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor, **ILU_OPTS)
        n = K.shape[0]
        return LinearOperator((n, n), matvec=lu.solve, dtype=float)
    return build


def coarse_space(ctx, n, kind, extend=True, theta=0.3, coarse='qbspline'):
    grid = ctx['grid']; nn = grid['nn']
    if kind == 'ms':
        B = build_basis(ctx['x'], ctx['Vp'], ctx['lam'], ctx['mu'], grid, n, fixed_nodes=ctx['fixed_nodes'], coarse=coarse)
        Phi, cg_ = B['Phi'], B['cg']
    elif kind == 'os':
        from common.msbasis_os import build_basis_os
        B = build_basis_os(ctx['x'], ctx['Vp'], ctx['lam'], ctx['mu'], grid, n, ctx['fixed_nodes'])
        Phi, cg_ = B['Phi'], B['cg']
    elif kind == 'piml':
        from ml.piml_basis import build_basis_piml
        B = build_basis_piml(ctx['model'], ctx['x'], ctx['Vp'], ctx['lam'], ctx['mu'], grid, n, ctx['fixed_nodes'])
        Phi, cg_ = B['Phi'], B['cg']
    else:
        cg_ = coarse_grid(grid, n)
        Pb = coarse_global(grid, cg_, np.arange(nn), kind).tolil(); Pb[ctx['fixed_nodes'], :] = 0
        Phi = sp.block_diag([Pb.tocsr()] * 2).tocsr()
    if extend:
        Phi, _ = extend_small_cut(Phi, cg_, np.concatenate([ctx['mi'], ctx['mi']]), theta)
    P = Phi[ctx['fdofs']].tocsc()
    if ctx.get('E') is not None:
        # 约化未知量 = 细尺度"好"自由度（E 的列）；粗空间取其在好自由度上的行（好节点处 E 为单位阵）
        E = ctx['E'].tocsc(); good_rows = np.asarray(E.argmax(0)).ravel()
        P = P[good_rows]
    nzc = np.where(np.asarray(abs(P).sum(0)).ravel() > 0)[0]
    return P[:, nzc].tocsr()


def twolevel_builder(n=4, kind='ms', drop_tol=1e-3, fill_factor=2.0, coarse='qbspline', P_cache=None, model=None):
    """P_cache：dict，可在多次调用间复用粗空间（例如同一载荷步内各 Newton 迭代）——由调用方决定何时失效。
    model：kind='piml' 时的 PIMLBasis 模型。"""
    def build(K, ctx):
        if model is not None:
            ctx = dict(ctx, model=model)
        T = ctx.get('timer')
        if T: T.tic('precond_basis')
        if P_cache is not None and P_cache.get('P') is not None and P_cache.get('key') == id(ctx['fdofs']):
            P = P_cache['P']
        else:
            P = coarse_space(ctx, n, kind, coarse=coarse)
            if P_cache is not None:
                P_cache['P'] = P; P_cache['key'] = id(ctx['fdofs'])
        if T: T.tic('precond_coarse')
        Kc = (P.T @ K @ P).tocsc()
        clu = splu(Kc, permc_spec='MMD_AT_PLUS_A')
        if T: T.tic('precond_ilu')
        lu = spilu(K.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor, **ILU_OPTS)
        if T: T.tic('precond')
        PT = P.T.tocsr()

        def apply(r):
            z = lu.solve(r)
            z = z + P @ clu.solve(PT @ (r - K @ z))
            z = z + lu.solve(r - K @ z, trans='T')
            return z
        m = K.shape[0]
        op = LinearOperator((m, m), matvec=apply, dtype=float)
        op.nc = P.shape[1]
        return op
    return build
