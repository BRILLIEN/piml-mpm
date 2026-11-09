# -*- coding: utf-8 -*-
"""
PIML 基网络：数据生成 + 训练 + 独立测试（全部问题无关的合成数据训练；测试另含来自悬臂梁模拟状态的真实粗单元）。
运行：GUARD_MAX=3000 tools/guard.sh python3 -m ml.train_basis --n 8 --ntrain 20000 --epochs 40
输出：ml/models/basis_n{n}.npz，ml/models/basis_n{n}_log.json（数据生成/训练耗时单独记录）
"""
import argparse, json, os, time
import numpy as np
from ml.piml_basis import random_images, features, targets, MLP, constraint_projector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--ntrain', type=int, default=20000)
    ap.add_argument('--ntest', type=int, default=2000)
    ap.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--coarse', default='qbspline')
    a = ap.parse_args()
    os.makedirs('ml/models', exist_ok=True)
    rng = np.random.default_rng(2026)
    log = dict(args=vars(a))
    # ---------------- 数据生成 ----------------
    t0 = time.perf_counter()
    il, im = random_images(a.ntrain + a.ntest, a.n, rng)
    t1 = time.perf_counter()
    Y = targets(il, im, a.n, a.coarse)                       # (N, 2ni, 32)
    t2 = time.perf_counter()
    X = features(il, im)
    log.update(t_images=t1 - t0, t_labels=t2 - t1)
    print(f'数据生成：图像 {t1-t0:.1f}s，精确标签 {t2-t1:.1f}s，样本 {len(X)}，输入维 {X.shape[1]}，输出维 {Y[0].size}')
    N = len(X); Yf = Y.reshape(N, -1)
    Xtr, Ytr = X[:a.ntrain], Yf[:a.ntrain]; Xte, Yte = X[a.ntrain:], Yf[a.ntrain:]
    xm = Xtr.mean(0); xs = Xtr.std(0) + 1e-8
    ys = np.full(Ytr.shape[1], Ytr.std())                      # 标量输出尺度（逐分量尺度会放大结构性零分量的噪声）
    Xtr_ = (Xtr - xm) / xs; Ytr_ = Ytr / ys; Xte_ = (Xte - xm) / xs
    Q = constraint_projector(a.n, a.coarse); ni2 = Y.shape[1]
    # ---------------- 训练（Adam，MSE；投影在损失中同样施加，保证训练/推理一致） ----------------
    net = MLP([X.shape[1], *a.hidden, Yf.shape[1]], seed=0)
    params = net.W + net.b
    mom = [np.zeros_like(p) for p in params]; vel = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8; step = 0
    t3 = time.perf_counter(); hist = []
    def proj(Yn):                                             # 归一化输出 -> 物理 ΔΦ -> 投影 -> 再归一化
        return ((Yn * ys).reshape(-1, ni2, 32) @ Q).reshape(len(Yn), -1) / ys
    for ep in range(a.epochs):
        lr = a.lr * (0.5 * (1 + np.cos(np.pi * ep / a.epochs)))      # 余弦退火
        perm = rng.permutation(a.ntrain); tot = 0.0
        for s in range(0, a.ntrain, a.batch):
            b = perm[s:s + a.batch]
            out, hs = net.forward(Xtr_[b], keep=True)
            pr = proj(out)
            r = pr - Ytr_[b]
            tot += np.sum(r ** 2)
            dY = proj(2 * r / len(b))                             # 投影为线性且对称（Q 对称），梯度同样投影
            gW, gb = net.backward(hs, dY)
            grads = gW + gb; step += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                mom[i] = b1 * mom[i] + (1 - b1) * g; vel[i] = b2 * vel[i] + (1 - b2) * g * g
                p -= lr * (mom[i] / (1 - b1 ** step)) / (np.sqrt(vel[i] / (1 - b2 ** step)) + eps)
        pte = proj(net.forward(Xte_)) * ys
        rel = np.linalg.norm(pte - Yte) / np.linalg.norm(Yte + 0)
        hist.append((ep, tot / a.ntrain, rel))
        print(f'  epoch {ep:3d}  train MSE(归一化) {tot / a.ntrain / Yf.shape[1]:.3e}  测试 ΔΦ 相对误差 {rel:.4f}', flush=True)
    t4 = time.perf_counter()
    log.update(t_train=t4 - t3, hist=hist, test_rel_err=float(hist[-1][2]), flops_per_cell=net.flops())
    meta = dict(n=a.n, coarse=a.coarse, xm=xm.tolist(), xs=xs.tolist(), ys=ys.tolist(), hidden=a.hidden)
    net.save(f'ml/models/basis_n{a.n}.npz', meta)
    json.dump(log, open(f'ml/models/basis_n{a.n}_log.json', 'w'), indent=1, default=float)
    print(f'训练耗时 {t4-t3:.1f}s；模型 -> ml/models/basis_n{a.n}.npz；每单元推理 {net.flops()/1e6:.2f} MFlop')


if __name__ == '__main__':
    main()
