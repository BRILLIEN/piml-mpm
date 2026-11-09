# -*- coding: utf-8 -*-
"""
问题无关机器学习（PIML）—— 学习"粗单元材料图像 → 多尺度局部基内部值 Φ_I"的局部映射。

学习对象（具有明确物理意义、可跨工况复用）
  输入：粗单元 n×n 细区间材料图像（λ_s, μ_s，含 ersatz 填充），只由局部材料决定，不含整体几何/载荷/边界信息。
  特征：f = [log(λ_s/ḡ), log(μ_s/ḡ)]，ḡ 为单元内 μ 的几何平均 —— Φ 对模量整体缩放不变（齐零次），故以比值为特征。
  输出：ΔΦ = Φ_I − W_I（相对粗 B 样条值的偏差），形状 (2ni, 32)。
物理约束（由构造精确满足，而非靠训练）
  单位分解与无穷小刚体转动：ΔΦ c_k = 0，c_k ∈ {x 平移, y 平移, 转动}（32 维粗自由度空间中的 3 个向量），
  通过正交投影 ΔΦ ← ΔΦ (I − C(CᵀC)⁻¹Cᵀ) 实现 ⇒ Φ = W + ΔΦ 精确保持 PU 与刚体再现（因 W 本身精确再现）。
网络：全连接 MLP（tanh），numpy 手写前向/反向 + Adam，可在 MATLAB 中等价实现（矩阵乘 + 逐元素函数）。
"""
import json, os, time
import numpy as np
from common.msbasis import local_layout, local_phiB, local_stiffness_images, solve_local, OFFS


# ----------------------------------------------------------------------------- 约束投影
def constraint_projector(n, coarse='qbspline'):
    """返回 (32,32) 投影矩阵 Q：ΔΦ Q 满足 ΔΦ c_k = 0。粗节点相对坐标 (cx, cy) ∈ {-1..2}²（以 H 为单位）。"""
    cx = np.array([c for cy in OFFS for c in OFFS], float); cy = np.array([c2 for c2 in OFFS for _ in OFFS], float)
    z = np.zeros(16); o = np.ones(16)
    C = np.stack([np.concatenate([o, z]), np.concatenate([z, o]), np.concatenate([-cy, cx])], 1)   # (32,3)
    return np.eye(32) - C @ np.linalg.solve(C.T @ C, C.T)


# ----------------------------------------------------------------------------- 数据生成（问题无关）
def random_images(N, n, rng, E_ref=1e6, nu_range=(0.25, 0.35), contrast_range=(1.0, 1000.0), ersatz=1e-3):
    """合成随机粗单元材料图像：随机圆形夹杂（数量、半径、位置、对比度）+ 随机直线切割（部分占有，模拟自由表面），
    每个细区间以 4x4 子采样计算体积分数。与任何具体整体问题无关。"""
    from common.material import lame
    sub = 4
    s = (np.arange(n * sub) + 0.5) / (n * sub)                       # 单元内归一化坐标 [0,1]
    SX, SY = np.meshgrid(s, s)
    il = np.empty((N, n, n)); im = np.empty((N, n, n))
    for k in range(N):
        nu_m = rng.uniform(*nu_range); nu_i = rng.uniform(*nu_range)
        Em = E_ref * np.exp(rng.uniform(-1, 1))
        con = np.exp(rng.uniform(np.log(contrast_range[0]), np.log(contrast_range[1])))
        Ei = Em * (con if rng.random() < 0.7 else 1.0 / con)          # 硬夹杂或软夹杂
        lm, mm = lame(Em, nu_m); li, mi = lame(Ei, nu_i)
        phase = np.zeros_like(SX, bool)
        for _ in range(rng.integers(0, 5)):
            r = rng.uniform(0.08, 0.45); xc = rng.uniform(-0.2, 1.2); yc = rng.uniform(-0.2, 1.2)
            phase |= (SX - xc) ** 2 + (SY - yc) ** 2 < r * r
        occ = np.ones_like(SX, bool)
        if rng.random() < 0.35:                                        # 自由表面切割
            th = rng.uniform(0, 2 * np.pi); d = rng.uniform(-0.6, 0.6)
            occ = (np.cos(th) * (SX - 0.5) + np.sin(th) * (SY - 0.5)) < d
        L = np.where(phase, li, lm) * occ; M = np.where(phase, mi, mm) * occ
        # 子采样 -> 区间平均（体积分数加权），空区间以 ersatz 填充（与 msbasis.cell_images 一致）
        phi = occ.reshape(n, sub, n, sub).mean((1, 3))
        Ls = L.reshape(n, sub, n, sub).sum((1, 3)); Ms = M.reshape(n, sub, n, sub).sum((1, 3))
        cnt = occ.reshape(n, sub, n, sub).sum((1, 3))
        lbar = np.where(cnt > 0, Ls / np.maximum(cnt, 1), 0); mbar = np.where(cnt > 0, Ms / np.maximum(cnt, 1), 0)
        lref = min(lm, li); mref = min(mm, mi)
        il[k] = phi * lbar + (1 - phi) * ersatz * lref; im[k] = phi * mbar + (1 - phi) * ersatz * mref
    return il, im


def features(il, im):
    g = np.exp(np.mean(np.log(im.reshape(len(im), -1)), 1))[:, None, None]
    return np.concatenate([np.log(il / g).reshape(len(il), -1), np.log(im / g).reshape(len(im), -1)], 1)


def targets(il, im, n, coarse='qbspline', chunk=2000):
    """精确局部凝聚得到 ΔΦ（训练标签）。"""
    _, _, WI = local_phiB(n, coarse)
    out = []
    for s in range(0, len(il), chunk):
        Kc = local_stiffness_images(il[s:s + chunk], im[s:s + chunk], n)
        out.append(solve_local(Kc, n, coarse) - WI[None])
    return np.concatenate(out, 0)


# ----------------------------------------------------------------------------- MLP（numpy）
class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X, keep=False):
        hs = [X]; h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            hs.append(h)
        return (h, hs) if keep else h

    def backward(self, hs, dY):
        gW = []; gb = []; d = dY
        for i in range(len(self.W) - 1, -1, -1):
            gW.insert(0, hs[i].T @ d); gb.insert(0, d.sum(0))
            if i > 0:
                d = (d @ self.W[i].T) * (1 - hs[i] ** 2)
        return gW, gb

    def flops(self):
        return 2 * sum(W.size for W in self.W)

    def save(self, path, meta):
        np.savez(path, *self.W, *self.b, meta=json.dumps(meta))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        k = len(d.files) - 1; L = k // 2
        net = MLP([1, 1]); net.W = [d[f'arr_{i}'] for i in range(L)]; net.b = [d[f'arr_{L + i}'] for i in range(L)]
        return net, json.loads(str(d['meta']))


class PIMLBasis:
    """训练好的模型 + 归一化 + 约束投影，提供批量预测 Φ_I（可直接替换 msbasis.solve_local）。"""
    def __init__(self, path):
        self.net, meta = MLP.load(path)
        self.meta = meta; self.n = meta['n']; self.coarse = meta['coarse']
        self.xm = np.array(meta['xm']); self.xs = np.array(meta['xs']); self.ys = np.array(meta['ys'])
        self.Q = constraint_projector(self.n, self.coarse)
        _, _, self.WI = local_phiB(self.n, self.coarse)

    def predict(self, il, im):
        X = (features(il, im) - self.xm) / self.xs
        Y = self.net.forward(X) * self.ys
        ni2 = self.WI.shape[0]
        dPhi = Y.reshape(len(X), ni2, 32) @ self.Q             # 精确投影：PU + 刚体转动
        return self.WI[None] + dPhi


def build_basis_piml(model, xp, Vp, lam, mu, grid, n, fixed_nodes=None, timer=None):
    """PIML 基：材料图像 → 网络预测 Φ_I；含 Dirichlet 节点的粗单元回退为精确局部凝聚。"""
    from common.msbasis import cell_images, basis_from_Kc, local_layout, local_stiffness_images
    assert model.n == n
    cax, cay, il, im, _ = cell_images(xp, Vp, lam, mu, grid, n)
    PhiI = model.predict(il, im)
    if fixed_nodes is not None:
        lay = local_layout(n, model.coarse); m = lay['m']
        isfix = np.zeros(grid['nn'], bool); isfix[fixed_nodes] = True
        lx = np.arange(m * m) % m; ly = np.arange(m * m) // m
        blk = isfix[(n * cax[:, None] + lx[None]) + (n * cay[:, None] + ly[None]) * grid['nx']]
        fb = np.where(blk.any(1))[0]                      # 回退单元
        if len(fb):
            sub = basis_from_Kc(local_stiffness_images(il[fb], im[fb], n), cax[fb], cay[fb], grid, n, fixed_nodes,
                                model.coarse, return_local=True)
            PhiI[fb] = sub['PhiI']
    out = basis_from_Kc(None, cax, cay, grid, n, fixed_nodes, model.coarse, PhiI=PhiI)
    out['n_fallback'] = int(len(fb)) if fixed_nodes is not None else 0
    return out
