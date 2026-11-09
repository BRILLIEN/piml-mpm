# -*- coding: utf-8 -*-
"""
多尺度数值形函数（粗尺度运动学基）—— 由粗单元内的粒子材料分布经局部凝聚得到。

构造（细网格 h 上的二次 B 样条节点，粗化比 n，H = n h）
--------------------------------------------------------
* 粗节点 A 位于细节点坐标 ξ_A = nA + 0.5（以 h 为单位、相对细网格原点）。
* 粗单元 A 覆盖 [nA+0.5, n(A+1)+0.5]；落在其中的粒子只与局部节点块 l ∈ [0, n+1]
  （全局细节点 nA + l）相互作用（二次 B 样条支撑 3 个节点）。
* 局部内部节点 l ∈ [2, n-1]（每方向 n-2 个），其刚度行**只**来自本粗单元内粒子；
  环带 l ∈ {0,1,n,n+1} 作为 Dirichlet 边界，取本单元粗双线性插值值（可外推半个 h）。
* 内部值：K_II Φ_I = −K_IB Φ_B（局部静力凝聚，线弹性，参数取粒子当前材料）。
* 全局：骨架节点（每个粗单元的环带内侧两层 l∈{1,n} 及其相邻单元同位置）取全局双线性值；
  内部节点取所在粗单元的局部解。

数学性质（由构造保证，见 test_msbasis.py 数值检验）
* 单位分解：Σ_J Φ_iJ = 1（各分量）。刚度行对常数场零化 + 边界值单位分解 ⇒ 内部单位分解。
* 无穷小刚体转动精确再现：线性边界值 + 零应变场为局部问题精确解。
* 均质粗单元内一次场精确再现；非均质单元内不再现（正是用来表征非均质性的部分）。
* 细尺度粒子插值仍用 C1 二次 B 样条 N_i(x)，v(x)=Σ_i N_i(x) Σ_J Φ_iJ V_J，
  因节点值单值，粒子层面的速度场仍 C1 连续 —— 粗化不重新引入 cell-crossing 梯度跳变。
"""
import numpy as np
import scipy.sparse as sp
from common.shape import shape_2d


def coarse_grid(grid, n):
    """粗网格：粗节点坐标与数目。"""
    h = grid['h']; ox, oy = grid['origin']
    ncx = (grid['nx'] - 1) // n + 1; ncy = (grid['ny'] - 1) // n + 1
    xs = ox + h * (n * np.arange(ncx) + 0.5); ys = oy + h * (n * np.arange(ncy) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    return dict(n=n, H=n * h, ncx=ncx, ncy=ncy, nc=ncx * ncy,
                xc=np.stack([X.ravel(), Y.ravel()], 1), origin=(xs[0], ys[0]))


def _f1d(t, kind):
    """粗尺度一维基函数（以粗节点间距无量纲化）：'bilinear' 帽函数 / 'qbspline' 中心二次 B 样条（C1）。"""
    a = np.abs(t)
    if kind == 'bilinear':
        return np.maximum(0.0, 1.0 - a)
    return np.where(a <= 0.5, 0.75 - a * a, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))


OFFS = np.array([-1, 0, 1, 2])          # 每个粗单元的边界值依赖 4x4 个粗节点（A-1..A+2）


def coarse_global(grid, cg, nodes, kind):
    """细节点处的全局粗尺度形函数（稀疏，(len(nodes), nc)）；kind='bilinear' 或 'qbspline'。"""
    n = cg['n']; i = nodes % grid['nx']; j = nodes // grid['nx']
    sx = (i - 0.5) / n; sy = (j - 0.5) / n
    Ax = np.floor(sx).astype(int); Ay = np.floor(sy).astype(int)
    rows, cols, vals = [], [], []
    for cy in OFFS:
        for cx in OFFS:
            jx = Ax + cx; jy = Ay + cy
            w = _f1d(sx - jx, kind) * _f1d(sy - jy, kind)
            ok = (w != 0) & (jx >= 0) & (jx < cg['ncx']) & (jy >= 0) & (jy < cg['ncy'])
            rows.append(np.arange(len(nodes))[ok]); cols.append((jx + jy * cg['ncx'])[ok]); vals.append(w[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(nodes), cg['nc']))


def bilinear_global(grid, cg, nodes):
    """细节点 nodes 处的全局粗双线性形函数，返回稀疏矩阵 (len(nodes), nc)。"""
    h = grid['h']; n = cg['n']
    i = nodes % grid['nx']; j = nodes // grid['nx']
    sx = (i - 0.5) / n; sy = (j - 0.5) / n
    ax = np.clip(np.floor(sx).astype(int), 0, cg['ncx'] - 2); ay = np.clip(np.floor(sy).astype(int), 0, cg['ncy'] - 2)
    fx = sx - ax; fy = sy - ay
    rows = np.repeat(np.arange(len(nodes)), 4)
    cols = np.stack([ax + ay * cg['ncx'], ax + 1 + ay * cg['ncx'],
                     ax + (ay + 1) * cg['ncx'], ax + 1 + (ay + 1) * cg['ncx']], 1).ravel()
    vals = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), cg['nc']))


def local_layout(n, kind='qbspline'):
    """局部节点块 (n+2)^2 的内部/边界划分，及块内节点处全局粗形函数值 W (m², 16)
    （列对应粗节点 (A+cx, A+cy)，cx,cy ∈ {-1,0,1,2}，顺序 cy 外层、cx 内层；与单元位置无关，所有单元共用）。"""
    m = n + 2
    lx, ly = np.meshgrid(np.arange(m), np.arange(m))
    lx = lx.ravel(); ly = ly.ravel()
    inner = (lx >= 2) & (lx <= n - 1) & (ly >= 2) & (ly <= n - 1)
    I = np.where(inner)[0]; B = np.where(~inner)[0]
    sx = (lx - 0.5) / n; sy = (ly - 0.5) / n          # 相对粗节点 A 的粗尺度坐标
    W = np.stack([_f1d(sx - cx, kind) * _f1d(sy - cy, kind) for cy in OFFS for cx in OFFS], 1)
    return dict(m=m, I=I, B=B, W=W, kind=kind)


def particle_cells(xp, grid, n):
    """粒子所属粗单元 (ax, ay) 及其 9 个支撑节点的局部编号。"""
    h = grid['h']; ox, oy = grid['origin']
    xi = (xp[:, 0] - ox) / h; eta = (xp[:, 1] - oy) / h
    ax = np.floor((xi - 0.5) / n).astype(np.int64); ay = np.floor((eta - 0.5) / n).astype(np.int64)
    bx = np.floor(xi - 0.5).astype(np.int64) - n * ax; by = np.floor(eta - 0.5).astype(np.int64) - n * ay
    return ax, ay, bx, by


def local_stiffness(xp, Vp, lam, mu, grid, n, cells_key, ax, ay, bx, by, chunk=200000):
    """按粗单元批量组装局部线弹性刚度 Kc (ncell, 2m², 2m²)。局部自由度顺序：[ux(m²), uy(m²)]。"""
    m = n + 2; m2 = m * m; nd = 2 * m2
    ncell = len(cells_key)
    _, inv = np.unique(ax + ay * 100003, return_inverse=True)   # 与 cells_key 对齐（调用方保证按同一规则排序）
    K = np.zeros(ncell * nd * nd)
    o = np.arange(3)
    for s in range(0, len(xp), chunk):
        e = slice(s, s + chunk)
        _, N, dNx, dNy = shape_2d(xp[e], grid, 'qbspline')
        lidx = ((bx[e, None, None] + o[None, None, :]) + (by[e, None, None] + o[None, :, None]) * m).reshape(-1, 9)
        V = Vp[e][:, None, None]; L = lam[e][:, None, None]; M = mu[e][:, None, None]
        gx_i = dNx[:, :, None]; gy_i = dNy[:, :, None]; gx_j = dNx[:, None, :]; gy_j = dNy[:, None, :]
        # 平面应变线弹性：K_ab = ∫ B_a^T D B_b
        Kxx = V * ((L + 2 * M) * gx_i * gx_j + M * gy_i * gy_j)
        Kxy = V * (L * gx_i * gy_j + M * gy_i * gx_j)
        Kyx = V * (L * gy_i * gx_j + M * gx_i * gy_j)
        Kyy = V * ((L + 2 * M) * gy_i * gy_j + M * gx_i * gx_j)
        base = inv[e][:, None, None] * (nd * nd)
        ri = lidx[:, :, None]; cj = lidx[:, None, :]
        for blk, (r0, c0) in ((Kxx, (0, 0)), (Kxy, (0, m2)), (Kyx, (m2, 0)), (Kyy, (m2, m2))):
            flat = (base + (ri + r0) * nd + (cj + c0)).ravel()
            K += np.bincount(flat, blk.ravel(), ncell * nd * nd)
    return K.reshape(ncell, nd, nd)


def raster_material(xp, Vp, lam, mu, grid):
    """把粒子材料栅格化到细尺度 B 样条节点区间（以节点 k 为中心、边长 h 的方块）：
    返回区间编号、体积占有率 φ 与体积加权 λ、μ。这是局部算子的"材料分布"输入。"""
    h = grid['h']; ox, oy = grid['origin']; nx = grid['nx']
    k = np.floor((xp[:, 0] - ox) / h + 0.5).astype(np.int64) + np.floor((xp[:, 1] - oy) / h + 0.5).astype(np.int64) * nx
    span, inv = np.unique(k, return_inverse=True)
    vol = np.bincount(inv, Vp); phi = np.minimum(vol / h ** 2, 1.0)
    lam_s = np.bincount(inv, Vp * lam) / vol; mu_s = np.bincount(inv, Vp * mu) / vol
    return span, phi, lam_s, mu_s


def gauss_points(span, phi, lam_s, mu_s, grid):
    """每个区间 3x3 Gauss 点（作为"伪粒子"），体积 = φ h² × 权重。"""
    h = grid['h']; xn = grid['xn']
    g = np.array([-0.5 * np.sqrt(0.6), 0.0, 0.5 * np.sqrt(0.6)]); w = np.array([5, 8, 5]) / 18.0
    GX, GY = np.meshgrid(g, g); WW = np.outer(w, w)
    xq = (xn[span][:, None, :] + h * np.stack([GX.ravel(), GY.ravel()], 1)[None]).reshape(-1, 2)
    Vq = (phi[:, None] * h ** 2 * WW.ravel()[None]).ravel()
    return xq, Vq, np.repeat(lam_s, 9), np.repeat(mu_s, 9)


def cell_images(xp, Vp, lam, mu, grid, n, ersatz=1e-3):
    """每个有粒子的粗单元生成 n×n 细区间"材料图像"（问题无关的局部输入）：
    λ_s = φ λ̄ + (1-φ) ε λ_ref，μ 同理；φ 为体积占有率，ε 为 ersatz 软材料比例（仅用于构造运动学基）。
    返回 (cax, cay, img_lam, img_mu, phi)，img 形状 (ncell, n, n)，[行=y, 列=x]。"""
    h = grid['h']; ox, oy = grid['origin']
    xi = (xp[:, 0] - ox) / h; eta = (xp[:, 1] - oy) / h
    ax = np.floor((xi - 0.5) / n).astype(np.int64); ay = np.floor((eta - 0.5) / n).astype(np.int64)
    sx = np.floor(xi - 0.5).astype(np.int64) - n * ax; sy = np.floor(eta - 0.5).astype(np.int64) - n * ay   # 区间局部号 0..n-1
    key = ax + ay * 100003
    keys, first, inv = np.unique(key, return_index=True, return_inverse=True)
    nc = len(keys); flat = inv * n * n + sy * n + sx
    vol = np.bincount(flat, Vp, nc * n * n)
    Ls = np.bincount(flat, Vp * lam, nc * n * n); Ms = np.bincount(flat, Vp * mu, nc * n * n)
    phi = np.minimum(vol / h ** 2, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        lbar = np.where(vol > 0, Ls / vol, 0.0); mbar = np.where(vol > 0, Ms / vol, 0.0)
    lref = np.min(lam); mref = np.min(mu)
    img_l = phi * lbar + (1 - phi) * ersatz * lref; img_m = phi * mbar + (1 - phi) * ersatz * mref
    return ax[first], ay[first], img_l.reshape(nc, n, n), img_m.reshape(nc, n, n), phi.reshape(nc, n, n)


def images_to_gauss(cax, cay, img_l, img_m, grid, n):
    """由材料图像生成 Gauss 伪粒子（每细区间 3x3 点，体积 h²·权重，满占）。区间 (sy,sx) 中心 = 细节点 nA+1+s。"""
    h = grid['h']; ox, oy = grid['origin']
    g = np.array([-0.5 * np.sqrt(0.6), 0.0, 0.5 * np.sqrt(0.6)]); w = np.array([5, 8, 5]) / 18.0
    GX, GY = np.meshgrid(g, g); WW = np.outer(w, w).ravel()
    s = np.arange(n)
    SX, SY = np.meshgrid(s, s)                                                # (n,n) [y,x]
    cx = ox + h * (n * cax[:, None, None] + 1 + SX[None])                     # 区间中心
    cy = oy + h * (n * cay[:, None, None] + 1 + SY[None])
    xq = np.stack([(cx[..., None] + h * GX.ravel()).ravel(), (cy[..., None] + h * GY.ravel()).ravel()], 1)
    nc = len(cax)
    Vq = np.tile(h * h * WW, nc * n * n)
    return xq, Vq, np.repeat(img_l.ravel(), 9), np.repeat(img_m.ravel(), 9)


_TPL = {}


def span_templates(n):
    """细区间刚度模板（与 h 无关：二维 ∫∇N·∇N dA 量纲为 1）。返回 (tλ, tμ, flat)：
    tλ, tμ 形状 (324,) 为单位 λ、单位 μ 时一个区间对其 3x3 节点（18 自由度）的刚度贡献；
    flat (n², 324) 为各区间贡献在粗单元局部矩阵 (2m²)² 中的展平位置。"""
    if n in _TPL:
        return _TPL[n]
    from common.shape import make_grid
    g = make_grid(0, 4, 0, 4, 1.0, pad=0)
    gq = np.array([-0.5 * np.sqrt(0.6), 0.0, 0.5 * np.sqrt(0.6)]); w = np.array([5, 8, 5]) / 18.0
    GX, GY = np.meshgrid(gq, gq); W = np.outer(w, w).ravel()
    xq = np.stack([2.0 + GX.ravel(), 2.0 + GY.ravel()], 1)          # 以节点 (2,2) 为中心的区间
    idx, N, dNx, dNy = shape_2d(xq, g, 'qbspline')
    assert np.all(idx == idx[0])
    gx = dNx; gy = dNy; V = W[:, None, None]
    gxi = gx[:, :, None]; gyi = gy[:, :, None]; gxj = gx[:, None, :]; gyj = gy[:, None, :]
    # λ 部分：B^T [[1,1,0],[1,1,0],[0,0,0]] B；μ 部分：B^T diag(2,2,1) B
    Lxx = (V * gxi * gxj).sum(0); Lxy = (V * gxi * gyj).sum(0); Lyx = (V * gyi * gxj).sum(0); Lyy = (V * gyi * gyj).sum(0)
    Mxx = (V * (2 * gxi * gxj + gyi * gyj)).sum(0); Mxy = (V * gyi * gxj).sum(0)
    Myx = (V * gxi * gyj).sum(0); Myy = (V * (2 * gyi * gyj + gxi * gxj)).sum(0)
    tl = np.concatenate([Lxx, Lxy, Lyx, Lyy], axis=None); tm = np.concatenate([Mxx, Mxy, Myx, Myy], axis=None)
    # 局部节点号：区间 (sy,sx) 中心局部节点 (sy+1, sx+1)，其 3x3 节点 (sy..sy+2, sx..sx+2)
    m = n + 2; m2 = m * m; nd = 2 * m2
    o = np.arange(3)
    s = np.arange(n); SX, SY = np.meshgrid(s, s); SX = SX.ravel(); SY = SY.ravel()
    loc = ((SX[:, None, None] + o[None, None, :]) + (SY[:, None, None] + o[None, :, None]) * m).reshape(n * n, 9)
    ri = loc[:, :, None]; cj = loc[:, None, :]
    blocks = [(0, 0), (0, m2), (m2, 0), (m2, m2)]
    flat = np.concatenate([((ri + r0) * nd + (cj + c0)).reshape(n * n, 81) for r0, c0 in blocks], 1)
    _TPL[n] = (tl, tm, flat)
    return _TPL[n]


def local_stiffness_images(img_l, img_m, n):
    """由材料图像直接组装粗单元局部刚度（模板法，无形函数求值）。返回 (ncell, nd, nd)。"""
    tl, tm, flat = span_templates(n)
    nc = img_l.shape[0]; m = n + 2; nd = 2 * m * m
    vals = img_l.reshape(nc, -1)[:, :, None] * tl[None, None, :] + img_m.reshape(nc, -1)[:, :, None] * tm[None, None, :]
    pos = (np.arange(nc)[:, None, None] * (nd * nd) + flat[None]).ravel()
    return np.bincount(pos, vals.ravel(), nc * nd * nd).reshape(nc, nd, nd)


def build_basis(xp, Vp, lam, mu, grid, n, return_local=False, quad='image', fixed_nodes=None, ersatz=1e-3, coarse='qbspline'):
    """构造全局多尺度基 Φ（稀疏，(2nn, 2nc)，自由度顺序 [ux 全部节点, uy 全部节点]）。
    quad='particle'：局部刚度直接用粒子积分（受粒子分布积分误差影响，一次场再现不精确）；
    quad='gauss'   ：先把粒子材料栅格化到细区间，再用 3x3 Gauss 积分（零阶积分一致，均质区一次场精确再现）。
    quad='image'   ：（默认）粗单元 n×n 材料图像（φ 加权 + ersatz 软材料填充空区间）+ Gauss 积分；
                     消除表面附近弱支撑节点导致的病态与基函数异常值。ersatz 只影响运动学基，不进入动力学内力。
    fixed_nodes    ：细尺度 Dirichlet（零位移）节点；在局部问题中作为零边界/零未知量施加，全局对应行置零，
                     使粗尺度空间精确满足细尺度位移边界条件。"""
    if quad == 'image':      # 推荐：粗单元材料图像（含 ersatz 正则化）+ 模板法局部刚度（等价于区间 3x3 Gauss 积分）
        cax, cay, il, im, _ = cell_images(xp, Vp, lam, mu, grid, n, ersatz)
        return basis_from_images(cax, cay, il, im, grid, n, fixed_nodes, coarse, return_local)
    elif quad == 'image_gauss':   # 同上但用 Gauss 伪粒子逐点组装（验证模板法用）
        cax0, cay0, il, im, _ = cell_images(xp, Vp, lam, mu, grid, n, ersatz)
        xp, Vp, lam, mu = images_to_gauss(cax0, cay0, il, im, grid, n)
    elif quad == 'gauss':
        xp, Vp, lam, mu = gauss_points(*raster_material(xp, Vp, lam, mu, grid), grid)
    cg = coarse_grid(grid, n)
    lay = local_layout(n, coarse)
    m = lay['m']; m2 = m * m; I = lay['I']; B = lay['B']
    ax, ay, bx, by = particle_cells(xp, grid, n)
    assert ax.min() >= 0 and ay.min() >= 0 and n * (ax.max() + 1) + 1 < grid['nx'] and n * (ay.max() + 1) + 1 < grid['ny'], \
        '粗单元局部节点块超出背景网格，请扩大网格 pad'
    keys, first = np.unique(ax + ay * 100003, return_index=True)
    cax = ax[first]; cay = ay[first]
    Kc = local_stiffness(xp, Vp, lam, mu, grid, n, keys, ax, ay, bx, by)
    return basis_from_Kc(Kc, cax, cay, grid, n, fixed_nodes, coarse, return_local)


def basis_from_images(cax, cay, il, im, grid, n, fixed_nodes=None, coarse='qbspline', return_local=False, PhiI=None):
    """由粗单元材料图像构造全局基（模板法组装 + 局部凝聚）；若给定 PhiI（如 PIML 网络预测），跳过局部求解。"""
    assert cax.min() >= 1 and cay.min() >= 1 and n * (cax.max() + 1) + 1 < grid['nx'] and n * (cay.max() + 1) + 1 < grid['ny'], \
        '粗单元局部节点块超出背景网格，请扩大网格 pad'
    Kc = None if PhiI is not None else local_stiffness_images(il, im, n)
    return basis_from_Kc(Kc, cax, cay, grid, n, fixed_nodes, coarse, return_local, PhiI=PhiI)


def local_phiB(n, coarse):
    lay = local_layout(n, coarse)
    nw = lay['W'].shape[1]; WB = lay['W'][lay['B']]; nb = len(lay['B'])
    PhiB = np.zeros((2 * nb, 2 * nw)); PhiB[:nb, :nw] = WB; PhiB[nb:, nw:] = WB
    WI = lay['W'][lay['I']]; ni = len(lay['I'])
    PhiI_c = np.zeros((2 * ni, 2 * nw)); PhiI_c[:ni, :nw] = WI; PhiI_c[ni:, nw:] = WI
    return lay, PhiB, PhiI_c


def solve_local(Kc, n, coarse='qbspline', fI2=None, fB2=None):
    """批量局部凝聚：返回 PhiI (ncell, 2ni, 32)。fI2/fB2 为固定（Dirichlet）内部/边界自由度掩码。"""
    lay, PhiB, PhiI_bil = local_phiB(n, coarse)
    m = lay['m']; m2 = m * m; I = lay['I']; B = lay['B']
    dI = np.concatenate([I, I + m2]); dB = np.concatenate([B, B + m2])
    KII = Kc[:, dI][:, :, dI]; KIB = Kc[:, dI][:, :, dB]
    diag = np.einsum('cii->ci', KII)
    scale = diag.max(1, keepdims=True)
    empty = diag < 1e-12 * scale
    KII = KII + np.einsum('ci,ij->cij', empty * scale + 1e-14 * scale, np.eye(len(dI)))
    if fI2 is not None:
        rhs = -KIB @ ((~fB2)[:, :, None] * PhiB[None])
        KII = KII * (~fI2)[:, :, None] * (~fI2)[:, None, :] + np.einsum('ci,ij->cij', fI2 * scale, np.eye(len(dI)))
        rhs = rhs * (~fI2)[:, :, None]
        empty = empty & ~fI2
    else:
        rhs = -KIB @ PhiB
    rhs = rhs + (empty * scale)[:, :, None] * PhiI_bil[None]
    return np.linalg.solve(KII, rhs)


def basis_from_Kc(Kc, cax, cay, grid, n, fixed_nodes=None, coarse='qbspline', return_local=False, PhiI=None):
    cg = coarse_grid(grid, n)
    lay = local_layout(n, coarse)
    m = lay['m']; m2 = m * m; I = lay['I']; B = lay['B']
    nxg = grid['nx']; ni = len(I); nn = grid['nn']; nc = cg['nc']
    fI2 = fB2 = None
    if fixed_nodes is not None:
        isfix = np.zeros(grid['nn'], bool); isfix[fixed_nodes] = True
        lxB = B % m; lyB = B // m; lxI = I % m; lyI = I // m
        fB = isfix[(n * cax[:, None] + lxB[None]) + (n * cay[:, None] + lyB[None]) * nxg]
        fI = isfix[(n * cax[:, None] + lxI[None]) + (n * cay[:, None] + lyI[None]) * nxg]
        fB2 = np.concatenate([fB, fB], 1); fI2 = np.concatenate([fI, fI], 1)
    if PhiI is None:
        PhiI = solve_local(Kc, n, coarse, fI2, fB2)
    elif fI2 is not None:
        PhiI = PhiI * (~fI2)[:, :, None]
    return _assemble_global(PhiI, cax, cay, grid, n, cg, lay, fixed_nodes, coarse, return_local, Kc)


def _unused_old_path(Kc, cax, cay, grid, n, fixed_nodes, coarse, lay, cg):
    m = lay['m']; m2 = m * m; I = lay['I']; B = lay['B']
    # 局部自由度：内部 dI，边界 dB
    dI = np.concatenate([I, I + m2]); dB = np.concatenate([B, B + m2])
    KII = Kc[:, dI][:, :, dI]; KIB = Kc[:, dI][:, :, dB]
    # 边界值：32 个粗自由度（4x4 粗节点 × 2 方向，顺序 [x(16), y(16)]），取全局粗形函数值（与全局 Φ 骨架行一致）
    nw = lay['W'].shape[1]
    WB = lay['W'][B]; nb = len(B)
    PhiB = np.zeros((2 * nb, 2 * nw)); PhiB[:nb, :nw] = WB; PhiB[nb:, nw:] = WB
    diag = np.einsum('cii->ci', KII)
    scale = diag.max(1, keepdims=True)
    empty = diag < 1e-12 * scale                       # 无粒子支撑的内部节点
    KII = KII + np.einsum('ci,ij->cij', empty * scale + 1e-14 * scale, np.eye(len(dI)))
    nxg = grid['nx']; ni = len(I)
    if fixed_nodes is not None:
        isfix = np.zeros(grid['nn'], bool); isfix[fixed_nodes] = True
        lxB = B % m; lyB = B // m; lxI = I % m; lyI = I // m
        fB = isfix[(n * cax[:, None] + lxB[None]) + (n * cay[:, None] + lyB[None]) * nxg]      # (ncell, nb)
        fI = isfix[(n * cax[:, None] + lxI[None]) + (n * cay[:, None] + lyI[None]) * nxg]      # (ncell, ni)
        fB2 = np.concatenate([fB, fB], 1); fI2 = np.concatenate([fI, fI], 1)
        rhs = -KIB @ ((~fB2)[:, :, None] * PhiB[None])
        # 固定的内部自由度：行列清零、对角置 scale、右端为零
        KII = KII * (~fI2)[:, :, None] * (~fI2)[:, None, :] + np.einsum('ci,ij->cij', fI2 * scale, np.eye(len(dI)))
        rhs = rhs * (~fI2)[:, :, None]
        empty = empty & ~fI2
    else:
        rhs = -KIB @ PhiB
    WI = lay['W'][I]
    PhiI_bil = np.zeros((2 * ni, 2 * nw)); PhiI_bil[:ni, :nw] = WI; PhiI_bil[ni:, nw:] = WI
    rhs = rhs + (empty * scale)[:, :, None] * PhiI_bil[None]   # 空节点取双线性值
    PhiI = np.linalg.solve(KII, rhs)                   # (ncell, 2ni, 32)
    return PhiI


def _assemble_global(PhiI, cax, cay, grid, n, cg, lay, fixed_nodes, coarse, return_local, Kc):
    m = lay['m']; I = lay['I']; ni = len(I)
    # ---- 组装全局 Φ ----
    nx = grid['nx']; nn = grid['nn']; nc = cg['nc']
    # 骨架行：只在粗单元局部块覆盖的细节点上计算全局粗形函数（其余节点不被任何粒子支撑，无需计算）
    blk = np.unique(((n * cax[:, None] + (np.arange(m * m) % m)[None]) +
                     (n * cay[:, None] + (np.arange(m * m) // m)[None]) * nx).ravel())
    Pb = coarse_global(grid, cg, blk, coarse).tocoo()
    Pb = sp.coo_matrix((Pb.data, (blk[Pb.row], Pb.col)), shape=(nn, nc))
    inner_rows = []; inner_cols = []; inner_vals = []
    lx = I % m; ly = I // m
    gnode = (n * cax[:, None] + lx[None]) + (n * cay[:, None] + ly[None]) * nx      # (ncell, ni)
    corners = np.stack([(cax + cx) + (cay + cy) * cg['ncx'] for cy in OFFS for cx in OFFS], 1)   # (ncell,16)
    ccol = np.concatenate([corners, corners + nc], 1)                                               # (ncell,32)
    nk = ccol.shape[1]
    for comp in range(2):
        rows = (gnode + comp * nn)[:, :, None].repeat(nk, 2)
        vals = PhiI[:, comp * ni:(comp + 1) * ni, :]
        cols = ccol[:, None, :].repeat(ni, 1)
        inner_rows.append(rows.ravel()); inner_cols.append(cols.ravel()); inner_vals.append(vals.ravel())
    inner_rows = np.concatenate(inner_rows); inner_cols = np.concatenate(inner_cols); inner_vals = np.concatenate(inner_vals)
    is_inner = np.zeros(2 * nn, bool); is_inner[inner_rows] = True
    # 双线性部分（x、y 两个分量块）
    br = np.concatenate([Pb.row, Pb.row + nn]); bc = np.concatenate([Pb.col, Pb.col + nc]); bv = np.concatenate([Pb.data, Pb.data])
    keep = ~is_inner[br]
    if fixed_nodes is not None:                       # 固定节点的骨架行置零
        fixrow = np.zeros(2 * nn, bool); fixrow[fixed_nodes] = True; fixrow[np.asarray(fixed_nodes) + nn] = True
        keep &= ~fixrow[br]
    Phi = sp.csr_matrix((np.concatenate([bv[keep], inner_vals]),
                         (np.concatenate([br[keep], inner_rows]), np.concatenate([bc[keep], inner_cols]))),
                        shape=(2 * nn, 2 * nc))
    Phi.eliminate_zeros()
    out = dict(Phi=Phi, cg=cg, ncell=len(cax))
    if return_local:
        out.update(PhiI=PhiI, cax=cax, cay=cay, Kc=Kc, lay=lay)
    return out


def extend_small_cut(Phi, cg, mh, theta=0.3, order=2):
    """小切割粗自由度的外推约束（扩展 B 样条思想，Höllig 2001 的 WEB-spline / extended B-spline）：
    粗节点 J 的"材料支撑度" d_J = Σ_i Φ_iJ² m_i（一致质量对角，两分量求和）；d_J < θ·d_ref（d_ref 为 90% 分位）
    视为小切割（坏）节点，令 V_J = Σ_K w_JK V_K，K 为距离最近的 (order+1)² 个好节点构成的规则块，
    w 为张量积 Lagrange 外推权（∑w=1，且再现 order 次多项式）⇒ Φ_eff = Φ C 保持单位分解、刚体与多项式再现，
    同时消除质量极小而刚度有限的自由度（其限制显式临界时间步、并使隐式粗算子病态）。
    返回 Phi_eff（坏节点列为零）与信息。"""
    nc = cg['nc']; ncx = cg['ncx']; ncy = cg['ncy']
    P2 = Phi.multiply(Phi).T.tocsr()
    d2 = P2 @ mh
    d = d2[:nc] + d2[nc:]
    pos = d > 1e-14 * d.max()
    if not pos.any():
        return Phi, dict(nbad=0)
    dref = np.percentile(d[pos], 90)
    good = d >= theta * dref
    bad = np.where(pos & ~good)[0]
    gi = good.reshape(ncy, ncx)
    k = order + 1
    # 规则块 (k×k) 全为好节点的左下角位置
    ok = np.ones((ncy - k + 1, ncx - k + 1), bool)
    for a in range(k):
        for b in range(k):
            ok &= gi[a:a + ncy - k + 1, b:b + ncx - k + 1]
    by, bx = np.where(ok)
    if len(bx) == 0 and order > 1:
        return extend_small_cut(Phi, cg, mh, theta, 1)
    cen_x = bx + 0.5 * (k - 1); cen_y = by + 0.5 * (k - 1)
    rows = []; cols = []; vals = []
    gidx = np.where(good)[0]
    rows.append(gidx); cols.append(gidx); vals.append(np.ones(len(gidx)))
    for J in bad:
        jx = J % ncx; jy = J // ncx
        q = np.argmin((cen_x - jx) ** 2 + (cen_y - jy) ** 2)
        x0 = bx[q]; y0 = by[q]
        nodes_x = x0 + np.arange(k); nodes_y = y0 + np.arange(k)
        def lag(xs, t):
            w = np.ones(k)
            for a in range(k):
                for b in range(k):
                    if a != b:
                        w[a] *= (t - xs[b]) / (xs[a] - xs[b])
            return w
        wx = lag(nodes_x, jx); wy = lag(nodes_y, jy)
        W = np.outer(wy, wx).ravel()
        K = (nodes_x[None, :] + nodes_y[:, None] * ncx).ravel()
        rows.append(np.full(len(K), J)); cols.append(K); vals.append(W)
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    C1 = sp.csr_matrix((v, (r, c)), shape=(nc, nc))
    C = sp.block_diag([C1, C1]).tocsr()
    return (Phi @ C).tocsr(), dict(nbad=len(bad), ngood=int(good.sum()), dref=dref)
