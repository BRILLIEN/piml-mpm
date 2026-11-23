# -*- coding: utf-8 -*-
"""
三维异质梁隐式 MPM 的基础核函数（由用户 MATLAB 程序移植，函数名与变量名保持一致）：
  setup_heterogeneous_beam, beam_basis, beam_transfer, beam_neo, beam_tangent, BasicMat_trace
数据布局与 MATLAB 一致：s['x'] 为 (3,np)，s['F'] 为 (3,3,np)，节点量为 (3,nf)，自由度编号 3*(node)+comp（0 基）。
只使用 MATLAB 中有直接对应的运算（ndgrid/reshape/pagemtimes≈matmul/accumarray≈bincount/sparse/chol）。
"""
import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------------- 几何与材料
def setup_heterogeneous_beam(p, root):
    """基于 Chen–Li 图 20 重建两排旋转椭圆柱（柱轴沿 z）。与 setup_heterogeneous_beam.m 一致。"""
    L = p['h'] * np.array(p['cells'], float)
    start = p['h'] * np.array(p['pad'], float)
    dims = np.array(p['cells']) + 2 * np.array(p['pad']) + 1
    ix, iy, iz = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')
    grid = dict(node=np.stack([ix.ravel(order='F'), iy.ravel(order='F'), iz.ravel(order='F')], 1), dims=dims)
    nd = grid['node']
    grid['fixed'] = (nd[:, 0] <= p['pad'][0] + 1) | (nd[:, 0] >= p['pad'][0] + p['cells'][0] - 1)
    ax = [(np.arange(0.5, n * p['ppc']) ) * (p['h'] / p['ppc']) for n in p['cells']]
    X, Y, Z = np.meshgrid(*ax, indexing='ij')
    x = X.ravel(order='F'); y = Y.ravel(order='F'); z = Z.ravel(order='F')
    npart = x.size
    s = dict(x=np.stack([x, y, z]) + start[:, None], F=np.repeat(np.eye(3)[:, :, None], npart, 2),
             ever=np.zeros(npart, bool))
    fig20 = np.loadtxt(f'{root}/figure20_ellipses.csv', delimiter=',', skiprows=1)
    assert fig20.shape == (40, 5)
    stretch = L[:2] / np.array([20.0, 2.0])
    inclusions = fig20.copy(); inclusions[:, :2] = fig20[:, :2] * stretch
    for j in range(40):
        q = fig20[j]; c, s_angle = np.cos(q[4]), np.sin(q[4])
        U, S_, _ = np.linalg.svd(np.diag(stretch) @ np.array([[c, -s_angle], [s_angle, c]]) @ np.diag(q[2:4]))
        inclusions[j, 2:5] = [S_[0], S_[1], np.arctan2(U[1, 0], U[0, 0])]
    phase = np.zeros(npart, np.uint8)
    for j in range(40):
        q = inclusions[j]; a, b = np.cos(q[4]), np.sin(q[4])
        ids = np.where((np.abs(x - q[0]) <= q[2]) & (np.abs(y - q[1]) <= q[2]))[0]
        dx = x[ids] - q[0]; dy = y[ids] - q[1]
        ids = ids[((a * dx + b * dy) / q[2]) ** 2 + ((-b * dx + a * dy) / q[3]) ** 2 <= 1]
        assert ids.size > 0 and np.all(phase[ids] == 0), 'Inclusion resolution or overlap'
        phase[ids] = j + 1
    grid['phase'] = phase
    grid['V'] = np.full(npart, (p['h'] / p['ppc']) ** 3)
    grid['E'] = np.where(phase > 0, p['Einclusion'], p['E']).astype(float)
    grid['mu'] = grid['E'] / (2 * (1 + p['nu']))
    grid['lambda'] = grid['E'] * p['nu'] / ((1 + p['nu']) * (1 - 2 * p['nu']))
    grid['mass'] = p['rho'] * grid['V']
    s['v'] = np.zeros((3, npart)); s['C'] = np.zeros((3, 3, npart))
    return s, grid, inclusions


# ----------------------------------------------------------------------------- 粒子分组与形函数
_OFF = np.arange(27)
_SHIFT = np.stack([_OFF % 3, (_OFF // 3) % 3, _OFF // 9])        # (3,27)，x 快变


def beam_basis(s, grid, p):
    """按共用 27 个节点的粒子分组（同一支撑单元）。每个 patch 是一批具有相同粒子数 nq 的单元。
    返回 patch 列表：id (nq,nc)，node (27,nc)，N (nq,27,nc)，G (3,nq,27,nc)＝∂N/∂X（步初梯度经 F0 拉回参考构形）。"""
    h = p['h']; dims = grid['dims']
    base = np.floor(s['x'] / h - 0.5).astype(np.int64)
    assert np.all(base >= 0) and np.all(base + 2 < dims[:, None]), 'Particle support left background'
    home = base[0] + dims[0] * (base[1] + dims[1] * base[2])
    order = np.argsort(home, kind='stable'); sh = home[order]
    first = np.flatnonzero(np.r_[True, np.diff(sh) != 0])
    counts = np.diff(np.r_[first, home.size])
    stride = np.array([1, dims[0], dims[0] * dims[1]])
    patch = []
    for nq in np.unique(counts):
        cells = np.where(counts == nq)[0]
        for st in range(0, cells.size, 4096):
            sel = cells[st:st + 4096]; nc = sel.size
            ids = order[np.arange(nq)[:, None] + first[sel][None, :]]          # (nq,nc)
            nodes = (stride @ _SHIFT)[:, None] + sh[first[sel]][None, :]       # (27,nc)
            W = []; Dv = []
            for k in range(3):
                t = (s['x'][k, ids] / h - base[k, ids])[:, None, :]            # (nq,1,nc)
                wk = np.concatenate([0.5 * (1.5 - t) ** 2, 0.75 - (t - 1) ** 2, 0.5 * (t - 0.5) ** 2], 1)
                dk = np.concatenate([t - 1.5, 2 * (1 - t), t - 0.5], 1) / h
                W.append(wk[:, _SHIFT[k], :]); Dv.append(dk[:, _SHIFT[k], :])
            N = W[0] * W[1] * W[2]
            grad = [Dv[0] * W[1] * W[2], W[0] * Dv[1] * W[2], W[0] * W[1] * Dv[2]]   # ∇_x (步初构形)
            F0 = s['F'][:, :, ids]                                                # (3,3,nq,nc)
            G = np.zeros((3, nq, 27, nc))
            for j in range(3):
                for k in range(3):
                    G[j] += F0[k, j][:, None, :] * grad[k]                        # ∂N/∂X_j = Σ_k F0_kj ∂N/∂x_k
            uniq, scat = np.unique(nodes.ravel(order='F'), return_inverse=True)
            patch.append(dict(id=ids, node=nodes, N=N, G=G, uniqueNode=uniq, scatter=scat))
    signature = (sh[first].copy(), counts.copy())
    return patch, signature


def beam_transfer(mode, patch, grid, s, A=None, vg=None, h=None):
    """全粒子二次样条传递。mode: 'p2g'（质量、APIC 动量）、'force'（P→节点内力）、'increment'（节点位移→ΔF 增量）、'g2p'。"""
    nf = grid['node'].shape[0]; npart = s['x'].shape[1]
    if mode == 'p2g':
        out = np.zeros(nf); vout = np.zeros((3, nf))
    elif mode == 'force':
        out = np.zeros((3, nf))
    elif mode == 'increment':
        out = np.zeros((3, 3, npart))
    elif mode == 'g2p':
        out = np.zeros((3, npart)); vout = np.zeros((3, npart)); Cout = np.zeros((3, 3, npart))
    for B in patch:
        ids = B['id']; nodes = B['node']; nq, nc = ids.shape
        if mode == 'increment':
            u = A[:, nodes]                                                        # (3,27,nc)
            out[:, :, ids] = np.einsum('ian,jqan->ijqn', u, B['G'])                # Σ_a u_ia G_ja
        elif mode == 'force':
            P = A[:, :, ids] * grid['V'][ids][None, None]                          # (3,3,nq,nc)
            force = np.einsum('ijqn,jqan->ian', P, B['G'])                         # (3,27,nc)
            for k in range(3):
                out[k, B['uniqueNode']] += np.bincount(B['scatter'], force[k].ravel(order='F'), B['uniqueNode'].size)
        elif mode == 'p2g':
            massN = B['N'] * grid['mass'][ids][:, None, :]                         # (nq,27,nc)
            Msum = massN.sum(0)
            out[B['uniqueNode']] += np.bincount(B['scatter'], Msum.ravel(order='F'), B['uniqueNode'].size)
            mom = np.einsum('kqn,qan->kan', s['v'][:, ids], massN)
            for j in range(3):
                dist = h * grid['node'][nodes, j][None] - s['x'][j, ids][:, None, :]
                mom += np.einsum('kqn,qan->kan', s['C'][:, j][:, ids], massN * dist)
            for k in range(3):
                vout[k, B['uniqueNode']] += np.bincount(B['scatter'], mom[k].ravel(order='F'), B['uniqueNode'].size)
        else:
            u = A[:, nodes]; v = vg[:, nodes]
            out[:, ids] = np.einsum('ian,qan->iqn', u, B['N'])
            vout[:, ids] = np.einsum('ian,qan->iqn', v, B['N'])
            for j in range(3):
                dist = h * grid['node'][nodes, j][None] - s['x'][j, ids][:, None, :]
                Cout[:, j][:, ids] = (4 / h ** 2) * np.einsum('ian,qan->iqn', v, B['N'] * dist)
    if mode == 'p2g':
        return out, vout
    if mode == 'g2p':
        return out, vout, Cout
    return out


def beam_neo(F, grid, energyOnly=False):
    """可压缩 Neo-Hookean：P = μF − (μ − λ lnJ)F^{-T}；T = F^{-T}；gamma = μ − λ lnJ；W 小应变用 log1p 余项。"""
    npart = F.shape[2]
    a = F.reshape(9, npart, order='F')                                             # 列主序：a[0]=F11,a[1]=F21,...
    cof = np.stack([a[4] * a[8] - a[5] * a[7], a[5] * a[6] - a[3] * a[8], a[3] * a[7] - a[4] * a[6],
                    a[7] * a[2] - a[8] * a[1], a[8] * a[0] - a[6] * a[2], a[6] * a[1] - a[7] * a[0],
                    a[1] * a[5] - a[2] * a[4], a[2] * a[3] - a[0] * a[5], a[0] * a[4] - a[1] * a[3]])
    J = np.sum(a[:3] * cof[:3], 0)
    mu = grid['mu']; lam = grid['lambda']
    if np.any(J <= 1e-8):
        return None, None, None, J, np.full(npart, np.inf), None
    out_P = out_T = gamma = sigma = None
    if not energyOnly:
        ell = np.log(J); c = mu - lam * ell; t = cof / J
        stress = mu * a - c * t
        out_P = stress.reshape(3, 3, npart, order='F'); out_T = t.reshape(3, 3, npart, order='F'); gamma = c
        Pm = out_P
        sig = np.einsum('ikn,jkn->ijn', Pm, F) / J                                 # σ = P Fᵀ / J
        r2 = np.sqrt(2)
        sigma = np.stack([sig[0, 0], sig[1, 1], sig[2, 2], r2 * sig[2, 1], r2 * sig[2, 0], r2 * sig[1, 0]])
    b = a.copy(); b[[0, 4, 8]] -= 1
    tr = b[0] + b[4] + b[8]
    sec = b[0] * b[4] + b[4] * b[8] + b[0] * b[8] - b[1] * b[3] - b[2] * b[6] - b[5] * b[7]
    third = b[0] * (b[4] * b[8] - b[5] * b[7]) + b[1] * (b[5] * b[6] - b[3] * b[8]) + b[2] * (b[3] * b[7] - b[4] * b[6])
    z = tr + sec + third; ell = np.log1p(z); rem = z - ell
    small = np.abs(z) < 1e-3; zs = z[small]
    rem[small] = zs ** 2 * (.5 + zs * (-1 / 3 + zs * (.25 + zs * (-.2 + zs * (1 / 6 + zs * (-1 / 7 + zs / 8))))))
    W = mu * (.5 * np.sum(b ** 2, 0) - sec - third + rem) + .5 * lam * ell ** 2
    return out_P, out_T, gamma, J, W, sigma


# ----------------------------------------------------------------------------- 一致切线（细尺度）
def beam_cell_tangent(B, T, gamma, grid):
    """一个 patch 内各支撑单元的 81×81 一致材料切线（自由度顺序 3*a+i）：
    K_(a,i),(b,k) = Σ_p V[ μ δ_ik G_a·G_b + λ H_ai H_bk + γ H_bi H_ak ]，H_ai = Σ_J T_iJ G_aJ。
    三项都写成单元内粒子维的批量矩阵乘（BLAS，对应 MATLAB pagemtimes）。"""
    ids = B['id']; nq, nc = ids.shape
    G = B['G']                                                                    # (3,nq,27,nc)
    Gt = np.ascontiguousarray(G.transpose(3, 1, 2, 0))                            # (nc,nq,27,3) = G_aJ
    Tt = np.ascontiguousarray(T[:, :, ids].transpose(3, 2, 0, 1))                 # (nc,nq,3,3) = T_iJ
    H = np.matmul(Gt, Tt.transpose(0, 1, 3, 2))                                   # (nc,nq,27,3): H_ai = Σ_J G_aJ T_iJ
    h = H.reshape(nc, nq, 81)
    V = grid['V'][ids].T                                                          # (nc,nq)
    wl = (V * grid['lambda'][ids].T)[:, :, None]; wg = (V * gamma[ids].T)[:, :, None]
    wm = (V * grid['mu'][ids].T)[:, :, None, None]
    ht = h.transpose(0, 2, 1)                                                     # (nc,81,nq)
    A = np.matmul(ht, h * wl)                                                     # λ：H_ai H_bk
    Dg = np.matmul(ht, h * wg).reshape(nc, 27, 3, 27, 3)                          # Σ wg H_ak' H_bi' 记为 [(a,i'),(b,k')]
    D = Dg.transpose(0, 1, 4, 3, 2).reshape(nc, 81, 81)                           # 交换 i 与 k ⇒ H_bi H_ak
    g2 = Gt.transpose(0, 2, 1, 3).reshape(nc, 27, nq * 3)                         # (nc,27,(q,J))
    Mg = np.matmul(g2, (Gt * wm).transpose(0, 1, 3, 2).reshape(nc, nq * 3, 27))   # μ G_a·G_b
    K4 = (A + D).reshape(nc, 27, 3, 27, 3)
    for i in range(3):
        K4[:, :, i, :, i] += Mg
    return K4.reshape(nc, 81, 81)


def beam_tangent_pattern(grid, active):
    """活动自由度上的 CSR 结构（二次 B 样条：节点耦合范围 ±2，即 5×5×5 模板，每行至多 3×125 列）。
    返回 dict(indptr, indices, posmap(nfree,3,125), amap, nfree)。posmap 给出 (行, 列分量, 模板偏移) 在 CSR 数据中的位置。"""
    dims = grid['dims']; nf = grid['node'].shape[0]
    amap = -np.ones(nf, np.int64); amap[active] = np.arange(active.size)
    nfree = 3 * active.size
    o = np.arange(125); d = np.stack([o % 5 - 2, (o // 5) % 5 - 2, o // 25 - 2], 1)     # (125,3)
    stride = np.array([1, dims[0], dims[0] * dims[1]])
    xyz = grid['node'][active]                                                      # (na,3)
    nb_xyz = xyz[:, None, :] + d[None]                                              # (na,125,3)
    ok = np.all((nb_xyz >= 0) & (nb_xyz < dims), 2)
    nb = np.where(ok, nb_xyz @ stride, 0)
    nba = np.where(ok, amap[nb], -1)                                                # 邻居活动序号
    # 列按 (邻居偏移 o 外层, 分量 k 内层) 排列：活动节点按网格编号递增、偏移按 (dz,dy,dx) 字典序 ⇒ 列号天然递增，无需排序
    cols = np.where(nba[:, :, None] >= 0, 3 * nba[:, :, None] + np.arange(3)[None, None, :], -1)   # (na,125,3)
    present = np.repeat((cols >= 0)[:, None], 3, 1).reshape(nfree, 375)                          # 行 = 3*node+i
    cnt = present.sum(1)
    indptr = np.r_[0, np.cumsum(cnt)]
    rank = np.cumsum(present, 1) - 1
    posmap = np.where(present, indptr[:-1, None] + rank, -1).astype(np.int32)                  # (nfree, 125*3) 顺序 (o,k)
    indices = np.repeat(cols[:, None], 3, 1).reshape(nfree, 375)[present].astype(np.int32)
    posmap = posmap.reshape(nfree, 125, 3).transpose(0, 2, 1).copy()                           # → (nfree, 3, 125)
    return dict(indptr=indptr, indices=indices, posmap=posmap, amap=amap, nfree=nfree,
                nnz=int(indptr[-1]))


def beam_tangent(patch, T, gamma, grid, pat, inertia, chunk=1024, use_cache=True):  # noqa
    """组装活动自由度上的有效切线 K_eff = K_tan + diag(inertia)（CSR，对称）。逐块（≤chunk 个支撑单元）积分并直接累加到 CSR 数据。"""
    dims = grid['dims']; amap = pat['amap']; posmap = pat['posmap']
    data = np.zeros(pat['nnz'])
    nodexyz = grid['node']
    cache = pat.setdefault('pos_cache', {}) if use_cache else {}               # 步内粒子分组不变 ⇒ 位置映射每步只算一次（仅在步内多次组装时有益）
    for ib, B in enumerate(patch):
        nc_all = B['id'].shape[1]
        for c0 in range(0, nc_all, chunk):
            cs = slice(c0, min(c0 + chunk, nc_all))
            Bc = dict(id=B['id'][:, cs], G=B["G"][..., cs])
            Ke = beam_cell_tangent(Bc, T, gamma, grid)                          # (nc,81,81)
            key = (ib, c0)
            if key not in cache:
                nodes = B['node'][:, cs]                                        # (27,nc)
                an = amap[nodes]
                xyz = nodexyz[nodes]                                            # (27,nc,3)
                dlt = xyz[None, :, :, :] - xyz[:, None, :, :] + 2               # (27a,27b,nc,3)：列节点 b 相对行节点 a
                off = dlt[..., 0] + 5 * dlt[..., 1] + 25 * dlt[..., 2]          # (27,27,nc)
                row = 3 * an[:, None, :] + np.arange(3)[None, :, None]          # (27,3,nc)
                valid = an >= 0
                r_ = np.where(valid[:, None, :], row, 0)
                pos = posmap[r_[:, :, None, None, :], np.arange(3)[None, None, None, :, None], off[:, None, :, None, :]]
                m = valid[:, None, None, None, :] & valid[None, None, :, None, :] & (pos >= 0)
                m = m.transpose(4, 0, 1, 2, 3).reshape(-1)                      # 与 Ke (nc,a,i,b,k) 内存顺序对齐
                pos = pos.transpose(4, 0, 1, 2, 3).reshape(-1)
                cache[key] = np.where(m, pos, pat['nnz']).astype(np.int32)       # 无效项指向哑槽 nnz（int32，省内存）
            ps = cache[key]
            if not use_cache:
                del cache[key]
            data += np.bincount(ps, Ke.reshape(-1), pat['nnz'] + 1)[:-1]
    K = sp.csr_matrix((data, pat['indices'], pat['indptr']), shape=(pat['nfree'], pat['nfree']))
    pat['last_data'] = data                                      # 不含惯性项的切线数据（供重叠块直接取子矩阵）
    return K + sp.diags(inertia)


def overlap_patches(grid, active, pat, inertia, stride=2, size=3):
    """重叠局部块：以偶数坐标节点为起点的 size³ 节点盒（stride=2 ⇒ 相邻块重叠 1 层，覆盖全部活动节点）。
    直接从结构化 CSR 数据按 (行, 分量, 模板偏移) 取 81×81 子矩阵（加惯性对角），批量求逆。
    返回 dict(dof (npatch, 3*size³) 活动自由度编号（缺失为 -1）, Kinv (npatch, n, n))。"""
    dims = grid['dims']; amap = pat['amap']; posmap = pat['posmap']; data = pat['last_data']
    xyz = grid['node'][active]
    starts = xyz[np.all(xyz % stride == 0, 1)]
    # 也包含被盒子截断的边缘：以每个活动节点向下取整到 stride 的起点
    starts = np.unique((xyz // stride) * stride, axis=0)
    o = np.arange(size ** 3); loc = np.stack([o % size, (o // size) % size, o // size ** 2], 1)
    nodes_xyz = starts[:, None, :] + loc[None]                                  # (np, size³, 3)
    inside = np.all(nodes_xyz < dims, 2)
    stride_v = np.array([1, dims[0], dims[0] * dims[1]])
    gnode = np.where(inside, np.minimum(nodes_xyz, dims - 1) @ stride_v, 0)
    an = np.where(inside, amap[gnode], -1)                                      # (np, m)
    keep = np.any(an >= 0, 1); an = an[keep]; nodes_xyz = nodes_xyz[keep]
    npch, m = an.shape
    dof = np.where(an[:, :, None] >= 0, 3 * an[:, :, None] + np.arange(3), -1).reshape(npch, 3 * m)
    d = nodes_xyz[:, None, :, :] - nodes_xyz[:, :, None, :] + 2                 # (np, a, b, 3) 列节点 b 相对行节点 a
    off = d[..., 0] + 5 * d[..., 1] + 25 * d[..., 2]
    row = np.where(an >= 0, 3 * an, 0)[:, :, None] + np.arange(3)                # (np, a, i)
    pos = posmap[row[:, :, :, None, None], np.arange(3)[None, None, None, None, :], off[:, :, None, :, None]]  # (np,a,i,b,k)
    ok = (an[:, :, None, None, None] >= 0) & (an[:, None, None, :, None] >= 0) & (pos >= 0)
    Kp = np.where(ok, data[np.maximum(pos, 0)], 0.0).reshape(npch, 3 * m, 3 * m)
    dg = np.where(dof >= 0, inertia[np.maximum(dof, 0)], 1.0)                    # 缺失自由度置单位对角
    Kp[:, np.arange(3 * m), np.arange(3 * m)] += dg
    return dict(dof=dof, Kinv=np.linalg.inv(Kp))


# ----------------------------------------------------------------------------- 三线性粗界面
def BasicMat_trace(node, active, fixed, dims, fine, shift=0):
    """二次粒子基支撑三节点，界面留相邻两层（mod(节点,fine)<2），使各块内部互不耦合。
    返回 dof_b（界面自由度，相对活动自由度编号）、dof_i（内部自由度）、N（界面细自由度 → 粗顶点自由度的三线性插值）。"""
    a = node[active]
    isb = np.any((a - shift) % fine < 2, 1)                     # shift：整体平移分块（用于错位第二分区）
    b = np.where(isb)[0]; i = np.where(~isb)[0]
    dof_b = (3 * b[:, None] + np.arange(3)).ravel(); dof_i = (3 * i[:, None] + np.arange(3)).ravel()
    x = a[b]; nr = x.shape[0]
    base = ((x - shift) // fine) * fine + shift; t = (x - base) / fine
    stride = np.array([1, dims[0], dims[0] * dims[1]])
    corner = np.array([[int(c) for c in f'{j:03b}'] for j in range(8)])
    col = np.zeros((nr, 8), np.int64); w = np.zeros((nr, 8))
    for j in range(8):
        v = corner[j]; w[:, j] = np.prod(t * v + (1 - t) * (1 - v), 1)
        zc = base + v * fine
        assert np.all((np.all(zc < dims, 1) & np.all(zc >= 0, 1)) | (w[:, j] == 0)), 'Need grid padding divisible by m'
        col[:, j] = np.clip(zc, 0, dims - 1) @ stride
    keep = (w > 0) & ~fixed[col]
    row = np.repeat(np.arange(nr)[:, None], 8, 1)
    Q = sp.csr_matrix((w[keep], (row[keep], col[keep])), shape=(nr, node.shape[0]))
    controls = np.where(np.asarray((Q != 0).sum(0)).ravel() > 0)[0]
    Q = Q[:, controls]
    # 截断导致的线性相关列：用列 Gram 矩阵的对角主元检查（与 MATLAB 版本 QR 去冗余等价的简化：去除近零范数列）
    N = sp.kron(Q, sp.identity(3)).tocsr()
    return dof_b, dof_i, N, b, i


def beam_cell_tangent_ref(B, T, gamma, grid):
    """一个 patch 内各支撑单元的 81×81 一致材料切线（自由度顺序 3*a+i）：
    K_(a,i),(b,k) = Σ_p V[ μ δ_ik G_a·G_b + λ H_ai H_bk + γ H_bi H_ak ]，H_ai = Σ_J T_iJ G_aJ。"""
    ids = B['id']; nq, nc = ids.shape
    G = B['G']                                                                    # (3,nq,27,nc)
    H = np.einsum('iJqn,Jqan->qnai', T[:, :, ids], G)                             # (nq,nc,27,3)
    V = grid['V'][ids]
    wl = V * grid['lambda'][ids]; wg = V * gamma[ids]; wm = V * grid['mu'][ids]   # (nq,nc)
    h = H.reshape(nq, nc, 81)                                                     # (3a+i)
    A = np.einsum('qn,qnr,qns->nrs', wl, h, h)                                    # λ 项
    D = np.einsum('qn,qnai,qnbk->naibk', wg, H, H)                                # γ：H_ai H_bk → 需要 H_bi H_ak
    D = D.transpose(0, 3, 2, 1, 4).reshape(nc, 81, 81)                            # (b,i,a,k)→ 行(a,i) 列(b,k) 取 H_bi H_ak
    Mg = np.einsum('qn,jqan,jqbn->nab', wm, G, G)                                 # μ G_a·G_b
    K = A + D
    K4 = K.reshape(nc, 27, 3, 27, 3)
    for i in range(3):
        K4[:, :, i, :, i] += Mg
    return K4.reshape(nc, 81, 81)


