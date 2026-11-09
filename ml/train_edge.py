# -*- coding: utf-8 -*-
"""
PIML-edge：学习"两单元材料图像 → 边自适应骨架条带值"（common/msbasis_os 第 2 步的局部块问题）。
净收益依据：精确两单元块问题（n=8：168 自由度稠密求解 + 40 右端 ≈ 4 MFlop/边）远贵于网络推理（≈0.75 MFlop/边）。
问题无关：训练数据为合成随机图像对（与悬臂梁等整体问题无关）；物理约束（PU/刚体转动）由投影精确满足。
运行：GUARD_MAX=3000 tools/guard.sh python3 -m ml.train_edge --n 8 --etype x --ntrain 20000 --epochs 30
"""
import argparse, json, os, time
import numpy as np
from ml.piml_basis import random_images, features, MLP
from common.msbasis_os import edge_layout, edge_solve_exact, coarse_vals


def edge_projector(CA, CB):
    k = len(CA); z = np.zeros(k); o = np.ones(k)
    C = np.stack([np.concatenate([o, z]), np.concatenate([z, o]), np.concatenate([-CB, CA]).astype(float)], 1)
    return np.eye(2 * k) - C @ np.linalg.solve(C.T @ C, C.T)


def strip_coarse(n, etype):
    """条带节点处粗函数值（相对坐标，边 (2,2)），形状 (2ns, 2ncol)。"""
    lay = edge_layout(n, etype); A0, B0 = 2, 2
    ox, oy = (n * A0, n * (B0 - 1)) if etype == 'x' else (n * (A0 - 1), n * B0)
    sn = lay['snodes']; gi = ox + lay['lx'][sn]; gj = oy + lay['ly'][sn]
    W = coarse_vals(gi, gj, n, A0 + lay['CA'][None], B0 + lay['CB'][None])
    ns = len(sn); k = len(lay['CA'])
    out = np.zeros((2 * ns, 2 * k)); out[:ns, :k] = W; out[ns:, k:] = W
    return out


def pair_images(N, n, etype, rng):
    il, im = random_images(2 * N, n, rng)
    ax = 0 if etype == 'x' else 1
    return np.concatenate([il[:N], il[N:]], ax + 1), np.concatenate([im[:N], im[N:]], ax + 1)


class PIMLEdge:
    def __init__(self, path):
        self.net, meta = MLP.load(path)
        self.n = meta['n']; self.etype = meta['etype']
        self.xm = np.array(meta['xm']); self.xs = np.array(meta['xs']); self.ys = float(meta['ys'])
        lay = edge_layout(self.n, self.etype)
        self.Q = edge_projector(lay['CA'], lay['CB']); self.W = strip_coarse(self.n, self.etype)

    def predict(self, IL, IM):
        X = (features(IL, IM) - self.xm) / self.xs
        Y = self.net.forward(X) * self.ys
        return self.W[None] + Y.reshape(len(X), *self.W.shape) @ self.Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--etype', default='x')
    ap.add_argument('--ntrain', type=int, default=20000)
    ap.add_argument('--ntest', type=int, default=2000)
    ap.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()
    rng = np.random.default_rng(7 if a.etype == 'x' else 8)
    t0 = time.perf_counter()
    IL, IM = pair_images(a.ntrain + a.ntest, a.n, a.etype, rng)
    t1 = time.perf_counter()
    Wst = strip_coarse(a.n, a.etype)
    Y = np.concatenate([edge_solve_exact(IL[s:s + 1000], IM[s:s + 1000], a.n, a.etype) for s in range(0, len(IL), 1000)]) - Wst[None]
    t2 = time.perf_counter()
    X = features(IL, IM); N = len(X); Yf = Y.reshape(N, -1)
    print(f'数据：图像 {t1-t0:.1f}s，精确标签 {t2-t1:.1f}s（{(t2-t1)/N*1e3:.3f} ms/边），输入 {X.shape[1]}，输出 {Yf.shape[1]}')
    Xtr, Ytr, Xte, Yte = X[:a.ntrain], Yf[:a.ntrain], X[a.ntrain:], Yf[a.ntrain:]
    xm = Xtr.mean(0); xs = Xtr.std(0) + 1e-8; ys = float(Ytr.std())
    Xtr_ = (Xtr - xm) / xs; Ytr_ = Ytr / ys; Xte_ = (Xte - xm) / xs
    lay = edge_layout(a.n, a.etype); Q = edge_projector(lay['CA'], lay['CB']); sh = Wst.shape
    proj = lambda Yn: (Yn.reshape(-1, *sh) @ Q).reshape(len(Yn), -1)
    net = MLP([X.shape[1], *a.hidden, Yf.shape[1]], seed=1)
    params = net.W + net.b; mo = [np.zeros_like(p) for p in params]; ve = [np.zeros_like(p) for p in params]; step = 0
    t3 = time.perf_counter(); hist = []
    for ep in range(a.epochs):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * ep / a.epochs)); perm = rng.permutation(a.ntrain); tot = 0.0
        for s in range(0, a.ntrain, a.batch):
            b = perm[s:s + a.batch]
            out, hs = net.forward(Xtr_[b], keep=True); r = proj(out) - Ytr_[b]; tot += np.sum(r ** 2)
            gW, gb = net.backward(hs, proj(2 * r / len(b))); step += 1
            for i, (p, g) in enumerate(zip(params, gW + gb)):
                mo[i] = 0.9 * mo[i] + 0.1 * g; ve[i] = 0.999 * ve[i] + 0.001 * g * g
                p -= lr * (mo[i] / (1 - 0.9 ** step)) / (np.sqrt(ve[i] / (1 - 0.999 ** step)) + 1e-8)
        pte = proj(net.forward(Xte_)) * ys
        rel = np.linalg.norm(pte - Yte) / np.linalg.norm(Yte); relS = np.linalg.norm(pte - Yte) / np.linalg.norm(Yte + Wst.ravel()[None])
        hist.append((ep, tot / a.ntrain, rel, relS))
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f'  epoch {ep:3d}  测试 Δ条带相对误差 {rel:.4f}  条带值相对误差 {relS:.5f}', flush=True)
    t4 = time.perf_counter()
    os.makedirs('ml/models', exist_ok=True)
    meta = dict(n=a.n, etype=a.etype, xm=xm.tolist(), xs=xs.tolist(), ys=ys, hidden=a.hidden)
    net.save(f'ml/models/edge_n{a.n}_{a.etype}.npz', meta)
    json.dump(dict(args=vars(a), t_images=t1 - t0, t_labels=t2 - t1, t_train=t4 - t3, hist=hist, flops=net.flops()),
              open(f'ml/models/edge_n{a.n}_{a.etype}_log.json', 'w'), indent=1, default=float)
    print(f'训练 {t4-t3:.1f}s；推理 {net.flops()/1e6:.2f} MFlop/边 -> ml/models/edge_n{a.n}_{a.etype}.npz')


if __name__ == '__main__':
    main()
