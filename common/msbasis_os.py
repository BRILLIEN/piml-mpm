# -*- coding: utf-8 -*-
"""
边自适应（两单元过采样）多尺度基 —— 解决 MsFEM 型"光滑骨架约束"在非均质（无尺度分离）时的系统性偏刚。

依据（数值证据，见 PROGRESS/报告）：全局静力凝聚（Guyan）中，若骨架整体强制为粗函数，非均质梁频率误差 11–16%；
若只强制粗节点处 2×2 角块、骨架自由，则误差降至 0.2–3%。故应让骨架随两侧材料自适应。

构造（细节点双线骨架、粗节点位于 2×2 角块中心，与 msbasis 相同的几何约定）
  1) 角块（粗节点处 2×2 细节点）：取全局粗二次 B 样条值。
  2) 骨架条带（每条粗边上的 2×(n−2) 个细节点）：在共享该边的两个粗单元构成的块上求局部问题
     （未知量 = 两单元内部 + 该条带；其余块内节点取粗函数值），取条带解。每条带只属于一条边 ⇒ 协调。
  3) 单元内部：以角块（粗函数）与四条自适应条带为 Dirichlet 边界做局部凝聚。
性质：各局部问题的边界数据满足单位分解且再现刚体 ⇒ Φ 精确单位分解、刚体再现（数值检验见 test）。
列（粗自由度）：单元内部依赖 6×6 个粗节点（72 列），比 msbasis 的 4×4（32 列）宽。
"""
import numpy as np
import scipy.sparse as sp
from common.msbasis import span_templates, _f1d, coarse_grid, coarse_global, cell_images
from common.shape import make_grid, shape_2d

_BT = {}


def block_templates(sny, snx):
    """矩形区间块（sny×snx 个区间，节点块 (sny+2)×(snx+2)）的模板展平位置。"""
    key = (sny, snx)
    if key in _BT:
        return _BT[key]
    tl, tm, _ = span_templates(2)
    my, mx = sny + 2, snx + 2; m2 = my * mx; nd = 2 * m2
    o = np.arange(3)
    SX, SY = np.meshgrid(np.arange(snx), np.arange(sny)); SX = SX.ravel(); SY = SY.ravel()
    loc = ((SX[:, None, None] + o[None, None, :]) + (SY[:, None, None] + o[None, :, None]) * mx).reshape(-1, 9)
    ri = loc[:, :, None]; cj = loc[:, None, :]
    flat = np.concatenate([((ri + r0) * nd + (cj + c0)).reshape(len(loc), 81)
                           for r0, c0 in [(0, 0), (0, m2), (m2, 0), (m2, m2)]], 1)
    _BT[key] = (tl, tm, flat, my, mx)
    return _BT[key]


def block_K_rows(il, im, sny, snx, rows):
    """只组装给定局部行（未知自由度）的块刚度 (nb, len(rows), nd)：避免整块 nd×nd 稠密分配。"""
    tl, tm, flat, my, mx = block_templates(sny, snx)
    nb = il.shape[0]; nd = 2 * my * mx; nr = len(rows)
    key = ('rows', sny, snx, nr, int(rows.sum()))
    if key not in _BT:
        rmap = -np.ones(nd, np.int64); rmap[rows] = np.arange(nr)
        r = flat // nd; c = flat % nd; ok = rmap[r] >= 0
        _BT[key] = (np.where(ok, rmap[r] * nd + c, 0), ok)
    fl2, ok = _BT[key]
    vals = (il.reshape(nb, -1)[:, :, None] * tl[None, None, :] + im.reshape(nb, -1)[:, :, None] * tm[None, None, :]) * ok[None]
    pos = (np.arange(nb)[:, None, None] * (nr * nd) + fl2[None]).ravel()
    return np.bincount(pos, vals.ravel(), nb * nr * nd).reshape(nb, nr, nd)


def block_K(il, im, sny, snx):
    tl, tm, flat, my, mx = block_templates(sny, snx)
    nb = il.shape[0]; nd = 2 * my * mx
    vals = il.reshape(nb, -1)[:, :, None] * tl[None, None, :] + im.reshape(nb, -1)[:, :, None] * tm[None, None, :]
    pos = (np.arange(nb)[:, None, None] * (nd * nd) + flat[None]).ravel()
    return np.bincount(pos, vals.ravel(), nb * nd * nd).reshape(nb, nd, nd)


def coarse_vals(gi, gj, n, cxs, cys):
    """全局细节点 (gi,gj) 处、粗节点 (cxs[k], cys[k]) 的粗二次 B 样条值。gi,gj:(...,)；返回 (..., K)。"""
    sx = (gi - 0.5) / n; sy = (gj - 0.5) / n
    return _f1d(sx[..., None] - cxs, 'qbspline') * _f1d(sy[..., None] - cys, 'qbspline')


def edge_layout(n, etype):
    """两单元块的局部布局：未知量掩码、条带节点、粗列偏移（相对边索引 (A,B)）。"""
    if etype == 'x':
        sny, snx = 2 * n, n
    else:
        sny, snx = n, 2 * n
    my, mx = sny + 2, snx + 2
    lx = np.arange(my * mx) % mx; ly = np.arange(my * mx) // mx
    if etype == 'x':
        inx = (lx >= 2) & (lx <= n - 1)
        unk_n = inx & (ly >= 2) & (ly <= 2 * n - 1); strip = inx & (ly >= n) & (ly <= n + 1)
        Aoff = np.arange(-1, 3); Boff = np.arange(-2, 3)
    else:
        iny = (ly >= 2) & (ly <= n - 1)
        unk_n = iny & (lx >= 2) & (lx <= 2 * n - 1); strip = iny & (lx >= n) & (lx <= n + 1)
        Aoff = np.arange(-2, 3); Boff = np.arange(-1, 3)
    CA, CB = np.meshgrid(Aoff, Boff)
    unk = np.concatenate([unk_n, unk_n]); U = np.where(unk)[0]
    sel = np.where(np.concatenate([strip, strip])[U])[0]
    return dict(sny=sny, snx=snx, my=my, mx=mx, lx=lx, ly=ly, unk=unk, sel=sel, snodes=np.where(strip)[0],
                CA=CA.ravel(), CB=CB.ravel())


def edge_images(ims, edges, n, etype):
    nE = len(edges); lay_s = (2 * n, n) if etype == 'x' else (n, 2 * n)
    IL = np.empty((nE,) + lay_s); IM = np.empty((nE,) + lay_s); org = np.empty((nE, 2), np.int64)
    for k, (A, B) in enumerate(edges):
        if etype == 'x':
            l0, m0 = ims.get(A, B - 1); l1, m1 = ims.get(A, B)
            IL[k] = np.concatenate([l0, l1], 0); IM[k] = np.concatenate([m0, m1], 0); org[k] = (n * A, n * (B - 1))
        else:
            l0, m0 = ims.get(A - 1, B); l1, m1 = ims.get(A, B)
            IL[k] = np.concatenate([l0, l1], 1); IM[k] = np.concatenate([m0, m1], 1); org[k] = (n * (A - 1), n * B)
    return IL, IM, org


def edge_solve_exact(IL, IM, n, etype, gi=None, gj=None, EA=None, EB=None, fl=None):
    """精确求两单元块问题，返回条带值 (nE, 2ns, 2ncol)。gi/gj 为块节点全局坐标（用于粗函数值）；
    若为 None（训练数据），取块原点 (n, n)·任意整数平移不变 —— 粗函数值只依赖相对位置。"""
    lay = edge_layout(n, etype); nE = IL.shape[0]
    K = block_K_rows(IL, IM, lay['sny'], lay['snx'], np.where(lay['unk'])[0])
    m2 = lay['my'] * lay['mx']; CA, CB = lay['CA'], lay['CB']; ncol = len(CA)
    if gi is None:                         # 相对坐标：边索引取 (A,B)=(2,2)，块原点相应
        A0, B0 = 2, 2
        ox, oy = (n * A0, n * (B0 - 1)) if etype == 'x' else (n * (A0 - 1), n * B0)
        gi = np.broadcast_to(ox + lay['lx'], (nE, m2)); gj = np.broadcast_to(oy + lay['ly'], (nE, m2))
        EA = np.full(nE, A0); EB = np.full(nE, B0); fl = np.zeros((nE, m2), bool)
    W = coarse_vals(gi, gj, n, (EA[:, None] + CA[None])[:, None, :], (EB[:, None] + CB[None])[:, None, :])
    Wbc = np.zeros((nE, 2 * m2, 2 * ncol)); Wbc[:, :m2, :ncol] = W; Wbc[:, m2:, ncol:] = W
    X = solve_block(K, lay['unk'], np.concatenate([fl, fl], 1), Wbc, rows_only=True)
    return X[:, lay['sel']]


class Images:
    """按粗单元索引取材料图像；无粒子的单元取 ersatz 图像。"""
    def __init__(self, cax, cay, il, im, ersatz_l, ersatz_m, n):
        self.map = {(a, b): k for k, (a, b) in enumerate(zip(cax.tolist(), cay.tolist()))}
        self.il = il; self.im = im; self.n = n
        self.el = np.full((n, n), ersatz_l); self.em = np.full((n, n), ersatz_m)

    def get(self, A, B):
        k = self.map.get((A, B))
        return (self.el, self.em) if k is None else (self.il[k], self.im[k])


def solve_block(K, unk, fixed_loc, Wbc, rows_only=False):
    """批量块问题：K (nb,nd,nd)（或 rows_only=True 时为只含未知行的 (nb,nu,nd)）；unk (nd,) 未知自由度掩码；
    fixed_loc (nb,nd) 固定自由度掩码；Wbc (nb,nd,nc) 边界自由度上的给定值。返回 (nb, nu, nc) 未知量的解。"""
    U = np.where(unk)[0]; Bd = np.where(~unk)[0]
    if rows_only:
        KUU = K[:, :, U]; KUB = K[:, :, Bd]
    else:
        KUU = K[:, U][:, :, U]; KUB = K[:, U][:, :, Bd]
    fU = fixed_loc[:, U]; fB = fixed_loc[:, Bd]
    diag = np.einsum('cii->ci', KUU); scale = diag.max(1, keepdims=True)
    empty = (diag < 1e-12 * scale) & ~fU
    rhs = -KUB @ (Wbc[:, Bd] * (~fB)[:, :, None])
    KUU = KUU * (~fU)[:, :, None] * (~fU)[:, None, :] + np.einsum('ci,ij->cij', (fU | empty) * scale + 1e-14 * scale, np.eye(len(U)))
    rhs = rhs * (~fU)[:, :, None] + (empty * scale)[:, :, None] * Wbc[:, U]
    return np.linalg.solve(KUU, rhs)


def build_basis_os(xp, Vp, lam, mu, grid, n, fixed_nodes=None, ersatz=1e-3, edge_models=None, timer=None):
    nx = grid['nx']; nn = grid['nn']; cg = coarse_grid(grid, n); nc = cg['nc']; ncx = cg['ncx']
    cax, cay, il, im, _ = cell_images(xp, Vp, lam, mu, grid, n, ersatz)
    ims = Images(cax, cay, il, im, ersatz * np.min(lam), ersatz * np.min(mu), n)
    isfix = np.zeros(nn, bool)
    if fixed_nodes is not None:
        isfix[fixed_nodes] = True
    occ = set(zip(cax.tolist(), cay.tolist()))
    rows = []; cols = []; vals = []
    strip_val = {}                       # (边类型, A, B) -> (全局节点, 值 (2*ns, 40), 列粗节点)
    # ---------------- 2) 骨架条带（x 向边：条带 j=nB,nB+1，位于单元 (A,B-1)/(A,B) 之间；y 向边同理） ----------------
    T = timer
    for etype in ('x', 'y'):
        edges = set()
        for (A, B) in occ:
            if etype == 'x':
                edges.add((A, B)); edges.add((A, B + 1))
            else:
                edges.add((A, B)); edges.add((A + 1, B))
        edges = sorted(edges)
        if T: T.tic('os_edge_img')
        IL, IM, org = edge_images(ims, edges, n, etype)
        lay = edge_layout(n, etype)
        EA = np.array([e[0] for e in edges]); EB = np.array([e[1] for e in edges])
        gi = org[:, 0:1] + lay['lx'][None]; gj = org[:, 1:2] + lay['ly'][None]
        gnode = gi + gj * nx
        fl = isfix[np.clip(gnode, 0, nn - 1)]
        if T: T.tic('os_edge_solve')
        if edge_models is not None and etype in edge_models:
            Xs = edge_models[etype].predict(IL, IM)                                  # (nE, 2ns, 2ncol)
            fb = np.where(fl.any(1))[0]                                              # 含固定节点 -> 精确回退
            if len(fb):
                Xs[fb] = edge_solve_exact(IL[fb], IM[fb], n, etype, gi[fb], gj[fb], EA[fb], EB[fb], fl[fb])
        else:
            Xs = edge_solve_exact(IL, IM, n, etype, gi, gj, EA, EB, fl)
        if T: T.tic('os_other')
        snodes = lay['snodes']; CA, CB = lay['CA'], lay['CB']
        for k, (A, B) in enumerate(edges):
            strip_val[(etype, A, B)] = (gnode[k, snodes], Xs[k],
                                        np.concatenate([(A + CA) + (B + CB) * ncx, (A + CA) + (B + CB) * ncx + nc]))
    # ---------------- 3) 单元内部：边界 = 角块（粗函数）+ 四条自适应条带 ----------------
    cells = sorted(occ); nC = len(cells)
    tl, tm, flat, my, mx = block_templates(n, n); m2 = my * mx
    IL = np.stack([ims.get(A, B)[0] for A, B in cells]); IM = np.stack([ims.get(A, B)[1] for A, B in cells])
    lx = np.arange(m2) % mx; ly = np.arange(m2) // mx
    inner = (lx >= 2) & (lx <= n - 1) & (ly >= 2) & (ly <= n - 1)
    K = block_K_rows(IL, IM, n, n, np.where(np.concatenate([inner, inner]))[0])
    CA, CB = np.meshgrid(np.arange(-2, 4), np.arange(-2, 4)); CA = CA.ravel(); CB = CB.ravel(); ncol = 36
    A_ = np.array([c[0] for c in cells]); B_ = np.array([c[1] for c in cells])
    gi = n * A_[:, None] + lx[None]; gj = n * B_[:, None] + ly[None]; gnode = gi + gj * nx
    W = coarse_vals(gi, gj, n, (A_[:, None] + CA[None])[:, None, :], (B_[:, None] + CB[None])[:, None, :])
    Wbc = np.zeros((nC, 2 * m2, 2 * ncol)); Wbc[:, :m2, :ncol] = W; Wbc[:, m2:, ncol:] = W
    colg = np.concatenate([(A_[:, None] + CA[None]) + (B_[:, None] + CB[None]) * ncx,
                           (A_[:, None] + CA[None]) + (B_[:, None] + CB[None]) * ncx + nc], 1)   # (nC, 72)
    # 用条带解覆盖环带上条带位置的边界值
    for k, (A, B) in enumerate(cells):
        cmap = {c: q for q, c in enumerate(colg[k].tolist())}
        pos = {g: q for q, g in enumerate(gnode[k].tolist())}
        for key in (('x', A, B), ('x', A, B + 1), ('y', A, B), ('y', A + 1, B)):
            gn, V, cl = strip_val[key]
            li = np.array([pos[g] for g in gn.tolist()])
            ci = np.array([cmap[c] for c in cl.tolist()])
            ns = len(gn)
            Wbc[k][li] = 0.0; Wbc[k][li + m2] = 0.0
            Wbc[k][np.ix_(li, ci)] = V[:ns]; Wbc[k][np.ix_(li + m2, ci)] = V[ns:]
    fl = isfix[gnode]; fixed_loc = np.concatenate([fl, fl], 1)
    unk = np.concatenate([inner, inner])
    X = solve_block(K, unk, fixed_loc, Wbc, rows_only=True)                       # (nC, 2ni, 72)
    ni = inner.sum(); inn = np.where(inner)[0]
    # ---------------- 全局组装：角块/其它骨架 = 粗函数；条带 = 边解；内部 = 单元解 ----------------
    assigned = np.zeros(2 * nn, bool)
    for comp in range(2):
        r = (gnode[:, inn] + comp * nn)[:, :, None].repeat(2 * ncol, 2)
        rows.append(r.ravel()); cols.append(np.broadcast_to(colg[:, None, :], r.shape).ravel())
        vals.append(X[:, comp * ni:(comp + 1) * ni, :].ravel())
        assigned[gnode[:, inn].ravel() + comp * nn] = True
    for (et, A, B), (gn, V, cl) in strip_val.items():
        ns = len(gn)
        for comp in range(2):
            r = np.repeat(gn + comp * nn, len(cl)); rows.append(r); cols.append(np.tile(cl, ns))
            vals.append(V[comp * ns:(comp + 1) * ns].ravel()); assigned[gn + comp * nn] = True
    blk = np.unique(gnode.ravel())
    Pb = coarse_global(grid, cg, blk, 'qbspline').tocoo()
    br = np.concatenate([blk[Pb.row], blk[Pb.row] + nn]); bc = np.concatenate([Pb.col, Pb.col + nc]); bv = np.concatenate([Pb.data, Pb.data])
    keep = ~assigned[br] & ~np.concatenate([isfix, isfix])[br]
    rows.append(br[keep]); cols.append(bc[keep]); vals.append(bv[keep])
    Phi = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * nn, 2 * nc))
    Phi.eliminate_zeros()
    return dict(Phi=Phi, cg=cg, ncell=nC, nedge=len(strip_val))
