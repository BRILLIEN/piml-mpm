# -*- coding: utf-8 -*-
"""
隐式 MPM 主程序：基础条件设置 --- 算例设置 --- 求解过程 --- 数据整理与比较
算例 I-beam：非均质悬臂梁在逐级增加的体力下的大变形准静态响应（几何与材料非线性）。
运行示例：
  tools/guard.sh python3 -m implicit.main_implicit --h 0.01 --nload 10 --solvers direct pcg_ilu pcg_2l_bil pcg_2l_ms
"""
import argparse, json, os, time
import numpy as np
from common.shape import make_grid
from common.material import lame
from common.hetero import random_inclusions, phase_of, particles_rect
from implicit.mpm_implicit import run_implicit
from implicit.precond import ilu_builder, twolevel_builder

# =====================================================================
# 一、基础条件设置
# =====================================================================
BASE = dict(
    L=1.0, H=0.2, h=0.01, ppc=2, rho=1050.0,
    E_m=1.0e6, nu_m=0.3, E_i=1.0e7, nu_i=0.3,           # 刚度对比 10
    frac=0.30, rmin=0.012, rmax=0.03, seed=1,
    gravity=(0.0, -20.0),                               # 最终体力（分 nload 步施加）
    nload=10, tol=1e-8, maxit=30, lin_tol=1e-8,
    fine_extend=0.05,                                   # 细尺度小支撑节点外推约束阈值（质量 < 0.05 满节点质量）
    n_coarse=4,                                         # 两级预条件粗化比
)


# =====================================================================
# 二、算例设置
# =====================================================================
def case_beam_static(cfg):
    L, H, h = cfg['L'], cfg['H'], cfg['h']
    X, Vp = particles_rect(0.0, L, 0.0, H, h, cfg['ppc'])
    incl = random_inclusions(0.0, L, 0.0, H, cfg['frac'], cfg['rmin'], cfg['rmax'], cfg['seed'])
    ph = phase_of(X, incl)
    lm, mm = lame(cfg['E_m'], cfg['nu_m']); li, mi = lame(cfg['E_i'], cfg['nu_i'])
    lam = np.where(ph == 1, li, lm); mu = np.where(ph == 1, mi, mm)
    Np = len(X)
    P = dict(x=X, m=np.full(Np, cfg['rho'] * Vp), V0=np.full(Np, Vp), lam=lam, mu=mu, phase=ph)
    # 网格原点使粗化比 2/4/8 的粗节点线落在 x=0（与显式路线同一约定）
    ox = -24.5 * h; oy = -0.7 * L - 20.5 * h
    nx = int(np.ceil((1.2 * L - ox) / h)) + 20; ny = int(np.ceil((H + 0.2 * L - oy) / h)) + 20
    grid = make_grid(ox + 3 * h, ox + 3 * h + (nx - 7) * h, oy + 3 * h, oy + 3 * h + (ny - 7) * h, h)
    fixed = np.where(grid['xn'][:, 0] <= 0.5 * h + 1e-9 * h)[0]
    mon = [int(np.argmin((X[:, 0] - L) ** 2 + (X[:, 1] - H / 2) ** 2)),
           int(np.argmin((X[:, 0] - 0.5 * L) ** 2 + (X[:, 1] - H / 2) ** 2))]
    par = dict(gravity=cfg['gravity'], nload=cfg['nload'], fixed_nodes=fixed, tol=cfg['tol'], maxit=cfg['maxit'],
               lin_tol=cfg['lin_tol'], monitor=mon, fine_extend=cfg['fine_extend'])
    return P, grid, par


# =====================================================================
# 三、求解过程
# =====================================================================
def solver_setup(name, cfg):
    """返回 (solver, precond_builder, 额外参数)。"""
    n = cfg['n_coarse']
    if name == 'direct':
        return 'direct', None, {}
    if name == 'pcg_ilu':
        return 'pcg', ilu_builder(), {}
    if name == 'pcg_2l_bil':
        return 'pcg', twolevel_builder(n, 'bilinear'), {}
    if name == 'pcg_2l_qb':
        return 'pcg', twolevel_builder(n, 'qbspline'), {}
    if name == 'pcg_2l_ms':
        return 'pcg', twolevel_builder(n, 'ms'), {}
    if name == 'pcg_2l_ms_reuse':      # 同一载荷步内复用粗空间（粒子—网格映射在步内冻结，材料分布不变）
        return 'pcg', twolevel_builder(n, 'ms', P_cache={}), {}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, default=BASE['h'])
    ap.add_argument('--nload', type=int, default=BASE['nload'])
    ap.add_argument('--n', type=int, default=BASE['n_coarse'])
    ap.add_argument('--g', type=float, default=BASE['gravity'][1])
    ap.add_argument('--solvers', nargs='+', default=['direct', 'pcg_ilu', 'pcg_2l_bil', 'pcg_2l_ms'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--fext', type=float, default=BASE['fine_extend'])
    ap.add_argument('--ls', action='store_true')
    a = ap.parse_args()
    cfg = dict(BASE, h=a.h, nload=a.nload, n_coarse=a.n, gravity=(0.0, a.g), fine_extend=a.fext)
    P, grid, par = case_beam_static(cfg)
    par['linesearch'] = a.ls
    tag = a.tag or f'imp_h{a.h}_n{a.n}_L{a.nload}'
    os.makedirs('results/implicit', exist_ok=True)
    out = dict(cfg={k: v for k, v in cfg.items()}, Np=len(P['m']), runs={})
    print(f'[I-beam] Np={len(P["m"])}，网格节点={grid["nn"]}，载荷步={a.nload}，粗化比={a.n}')
    xref = None
    for name in a.solvers:
        solver, pb, extra = solver_setup(name, cfg)
        print(f'===== {name} =====')
        t0 = time.perf_counter()
        R = run_implicit(P, grid, dict(par, **extra), solver=solver, precond_builder=pb, verbose=not a.quiet)
        wall = time.perf_counter() - t0
        print(R['timer'].report(name))
        H = R['hist']
        if xref is None:
            xref = R['x']
        err = float(np.sqrt(np.mean(np.sum((R['x'] - xref) ** 2, 1))))
        out['runs'][name] = dict(wall=wall, timing=R['timer'].t, newton=H['newton'].tolist(), lin_it=H['lin_it'].tolist(),
                                 ndof=H['ndof'].tolist(), tip=H['tip'].tolist(), Es=H['Es'].tolist(), x_rms_vs_first=err)
        print(f'  总耗时 {wall:.2f}s，Newton 总数 {H["newton"].sum()}，线性迭代总数 {H["lin_it"].sum()}，'
              f'自由端最终 y = {H["tip"][-1][1]:.6f}，与首个求解器的粒子位置 RMS 差 {err:.2e}')
        np.savez(f'results/implicit/{tag}_{name}.npz', x=R['x'], s=R['s'], F=R['F'])
    # =================================================================
    # 四、数据整理
    # =================================================================
    json.dump(out, open(f'results/implicit/{tag}.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('保存 ->', f'results/implicit/{tag}.json')


if __name__ == '__main__':
    main()
