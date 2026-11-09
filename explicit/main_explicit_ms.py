# -*- coding: utf-8 -*-
"""
显式路线对比主程序：基础条件设置 --- 算例设置 --- 求解过程 --- 数据整理与比较
同一算例、同一细网格与粒子：
  (a) 细尺度 USL 基准（mpm_explicit）
  (b) 多尺度粗运动学 + 区间冻结映射（mpm_explicit_ms，basis='ms'）
  (c) 对照：纯粗双线性运动学（basis='bilinear'，不含非均质信息）
时间步：各方法取各自临界时间步的同一比例（细尺度 CFL 选取的 dt 与其临界值之比），保证公平。
运行：tools/guard.sh python3 -m explicit.main_explicit_ms --h 0.01 --tend 0.4 --n 4 8 --K 25
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
from explicit.main_explicit import BASE, CASES
from explicit.mpm_explicit import run_explicit
from explicit.mpm_explicit_ms import run_explicit_ms
from common.msbasis import build_basis, coarse_grid, coarse_global, extend_small_cut
from common.assemble import elastic_K, lumped_mass, max_eig


def critical_ratio(P, grid, par, n, basis, extend=True, theta=0.3):
    """返回 λ_max(细)/λ_max(粗) 的平方根，即临界时间步放大倍数（t=0 构形，线弹性）。"""
    K = elastic_K(P['x'], P['V0'], P['lam'], P['mu'], grid); M = lumped_mass(P['x'], P['m'], grid)
    nn = grid['nn']; fx = np.zeros(2 * nn, bool); fx[par['fixed_nodes']] = True; fx[par['fixed_nodes'] + nn] = True
    lf = max_eig(K, M, (~fx) & (M > 1e-12 * M.max()))
    if basis == 'ms':
        B = build_basis(P['x'], P['V0'], P['lam'], P['mu'], grid, n, fixed_nodes=par['fixed_nodes'])
    elif basis == 'os':
        from common.msbasis_os import build_basis_os
        B = build_basis_os(P['x'], P['V0'], P['lam'], P['mu'], grid, n, fixed_nodes=par['fixed_nodes'])
    else:
        cg = coarse_grid(grid, n); Pb = coarse_global(grid, cg, np.arange(nn), 'bilinear').tolil(); Pb[par['fixed_nodes'], :] = 0
        Pb = Pb.tocsr(); B = dict(Phi=sp.block_diag([Pb, Pb]).tocsr(), cg=cg)
    Phi = B['Phi']
    if extend:
        Phi, _ = extend_small_cut(Phi, B['cg'], M, theta)
    Kc = (Phi.T @ K @ Phi).tocsr(); Mc = (Phi.T @ sp.diags(M) @ Phi).tocsr()
    d = Mc.diagonal(); fc = d > 1e-10 * d.max()
    lc = max_eig(Kc, Mc, fc)
    return float(np.sqrt(lf / lc)), float(lf), float(lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='beam')
    ap.add_argument('--h', type=float, default=0.01)
    ap.add_argument('--tend', type=float, default=0.4)
    ap.add_argument('--n', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--K', type=int, nargs='+', default=[25])
    ap.add_argument('--basis', nargs='+', default=['ms', 'bilinear'])
    ap.add_argument('--dtfac', type=float, default=1.0, help='粗尺度时间步相对"同比例临界步"的额外系数')
    ap.add_argument('--skip_fine', action='store_true')
    ap.add_argument('--theta', type=float, default=0.3, help='小切割外推阈值（0 关闭）')
    ap.add_argument('--flip_ms', type=float, default=0.0, help='多尺度方法区间末粒子速度更新的 FLIP 比例（默认 0：PIC，仅在区间交界处传递一次）')
    ap.add_argument('--out_every', type=int, default=20)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    # ---------- 一、基础条件 / 二、算例 ----------
    cfg = dict(BASE, h=a.h)
    P, grid, par, mon, incl = CASES[a.case](cfg)
    os.makedirs('results/explicit', exist_ok=True)
    tag = a.tag or f'ms_{a.case}_h{a.h}_t{a.tend}'
    dt0 = par['dt']
    out = dict(case=a.case, h=a.h, Np=len(P['m']), tend=a.tend, runs={})
    # ---------- 三、求解过程 ----------
    if not a.skip_fine:
        par_f = dict(par, nstep=int(round(a.tend / dt0)), out_every=a.out_every)
        print(f'[细尺度基准] Np={len(P["m"])} dt={dt0:.3e} nstep={par_f["nstep"]}')
        R = run_explicit(P, grid, par_f, monitor=mon)
        print(R['timer'].report('fine'))
        np.savez(f'results/explicit/{tag}_fine.npz', x=R['x'], **{'h_' + k: v for k, v in R['hist'].items()})
        out['runs']['fine'] = dict(dt=dt0, nstep=par_f['nstep'], timing=R['timer'].t, t_total=R['timer'].total())
    for basis in a.basis:
        for n in a.n:
            ratio, lf, lc = critical_ratio(P, grid, par, n, basis, extend=a.theta > 0, theta=a.theta)
            dt = dt0 * ratio * a.dtfac
            nstep = int(round(a.tend / dt)); dt = a.tend / nstep
            out_every = max(1, int(round(a.out_every * dt0 / dt)))
            for K in a.K:
                name = f'{basis}_n{n}_K{K}'
                print(f'[{name}] 临界步放大 {ratio:.2f}，dt={dt:.3e}，nstep={nstep}')
                par_r = dict(par, dt=dt, nstep=nstep, out_every=out_every, flip=a.flip_ms, extend=a.theta > 0, theta=a.theta)
                try:
                    R = run_explicit_ms(P, grid, par_r, n=n, K_int=K, basis=basis, monitor=mon)
                except FloatingPointError as e:
                    print('  失败：', e); out['runs'][name] = dict(failed=str(e)); continue
                print(R['timer'].report(name))
                np.savez(f'results/explicit/{tag}_{name}.npz', x=R['x'], **{'h_' + k: v for k, v in R['hist'].items() if k != 'n_int'})
                out['runs'][name] = dict(dt=dt, nstep=nstep, dt_ratio=ratio, lam_f=lf, lam_c=lc, K=K, n=n, basis=basis,
                                         n_int=R['hist']['n_int'], nc=R['nc'], timing=R['timer'].t, t_total=R['timer'].total())
    # ---------- 四、数据整理 ----------
    json.dump(out, open(f'results/explicit/{tag}.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('保存 ->', f'results/explicit/{tag}.json')


if __name__ == '__main__':
    main()
