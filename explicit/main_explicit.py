# -*- coding: utf-8 -*-
"""
显式 MPM 主程序：基础条件设置 --- 算例设置 --- 求解过程 --- 数据整理与比较
运行示例（必须经截止守卫，单线程）：
  tools/guard.sh python3 -m explicit.main_explicit --case beam --h 0.01 --nstep 2000
"""
import argparse, json, os, time
import numpy as np
from common.shape import make_grid
from common.material import lame
from common.hetero import random_inclusions, phase_of, particles_rect
from explicit.mpm_explicit import run_explicit

# =====================================================================
# 一、基础条件设置（集中设置：尺度、网格、粒子、材料、时间积分）
# =====================================================================
BASE = dict(
    L=1.0, H=0.2,                 # 悬臂梁长度、高度 [m]
    h=0.01,                       # 背景网格间距 [m]
    ppc=2,                        # 每单元每方向粒子数（ppc x ppc）
    rho=1050.0,                   # 密度 [kg/m3]（两相相同，仅刚度非均质）
    E_m=1.0e6, nu_m=0.3,          # 基体
    E_i=1.0e7, nu_i=0.3,          # 夹杂（刚度对比 10）
    frac=0.30, rmin=0.012, rmax=0.03, seed=1,   # 随机圆形夹杂
    gravity=(0.0, -10.0),         # 体力加速度 [m/s2]（强动力加载，使梁产生大挠度）
    cfl=0.4,                      # CFL 数
    flip=0.99,                    # FLIP 比例
    shape='qbspline',             # 'qbspline' 或 'linear'
    grid_shift=(0.0, 0.0),        # 背景网格整体偏移（以 h 为单位），用于 cell-crossing 对比
)


# =====================================================================
# 二、算例设置
# =====================================================================
def case_beam(cfg):
    """算例 E-beam：左端固支的非均质悬臂梁，在体力作用下由静止释放，产生大幅振动。
    特点：大量物质点、显著粒子迁移（末端位移 ~ 0.3L）、有限变形、材料非均质。"""
    L, H, h = cfg['L'], cfg['H'], cfg['h']
    X, Vp = particles_rect(0.0, L, 0.0, H, h, cfg['ppc'])
    incl = random_inclusions(0.0, L, 0.0, H, cfg['frac'], cfg['rmin'], cfg['rmax'], cfg['seed'])
    ph = phase_of(X, incl)
    lm, mm = lame(cfg['E_m'], cfg['nu_m']); li, mi = lame(cfg['E_i'], cfg['nu_i'])
    lam = np.where(ph == 1, li, lm); mu = np.where(ph == 1, mi, mm)
    Np = len(X)
    P = dict(x=X, v=np.zeros((Np, 2)), m=np.full(Np, cfg['rho'] * Vp), V0=np.full(Np, Vp),
             lam=lam, mu=mu, phase=ph)
    sx, sy = cfg['grid_shift']
    # 背景网格：原点取 ox = -(8*3+0.5)h，使粗化比 n∈{2,4,8} 时粗节点线恰位于固支面 x=0；
    # 范围覆盖梁的运动包络（x ≤ 1.2L，y ∈ [-0.7L, H+0.2L]），四周留足粗单元局部节点块所需的边。
    ox = -24.5 * h + sx * h; oy = -0.7 * L - 20.5 * h + sy * h
    nx = int(np.ceil((1.2 * L - ox) / h)) + 20; ny = int(np.ceil((H + 0.2 * L - oy) / h)) + 20
    grid = make_grid(ox + 3 * h, ox + 3 * h + (nx - 7) * h, oy + 3 * h, oy + 3 * h + (ny - 7) * h, h)
    fixed = np.where(grid['xn'][:, 0] <= 0.75 * h + 1e-9 * h)[0]  # x<=0.75h：任意网格偏移下至少两列节点固定 -> 固支（偏移 0 时即 -0.5h, 0.5h）
    cmax = np.sqrt((max(li, lm) + 2 * max(mi, mm)) / cfg['rho'])
    dt = cfg['cfl'] * h / cmax
    # 监测粒子：自由端中点、固支端上下表面附近（根部应力最大）
    mon = [int(np.argmin((X[:, 0] - L) ** 2 + (X[:, 1] - H / 2) ** 2)),
           int(np.argmin((X[:, 0] - 0.05 * L) ** 2 + (X[:, 1] - H) ** 2)),
           int(np.argmin((X[:, 0] - 0.5 * L) ** 2 + (X[:, 1] - H / 2) ** 2))]
    return P, grid, dict(dt=dt, gravity=cfg['gravity'], flip=cfg['flip'], fixed_nodes=fixed), mon, incl


CASES = dict(beam=case_beam)


# =====================================================================
# 三、求解过程
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='beam')
    ap.add_argument('--h', type=float, default=BASE['h'])
    ap.add_argument('--ppc', type=int, default=BASE['ppc'])
    ap.add_argument('--nstep', type=int, default=None)
    ap.add_argument('--tend', type=float, default=None)
    ap.add_argument('--shape', default=BASE['shape'])
    ap.add_argument('--shift', type=float, nargs=2, default=BASE['grid_shift'])
    ap.add_argument('--out_every', type=int, default=50)
    ap.add_argument('--tag', default='')
    ap.add_argument('--scheme', default='USL', help='USL 或 MUSL')
    a = ap.parse_args()
    cfg = dict(BASE, h=a.h, ppc=a.ppc, shape=a.shape, grid_shift=tuple(a.shift))
    t0 = time.perf_counter()
    P, grid, par, mon, incl = CASES[a.case](cfg)
    nstep = a.nstep if a.nstep is not None else int(np.ceil(a.tend / par['dt']))
    par.update(nstep=nstep, out_every=a.out_every, scheme=a.scheme)
    t_setup = time.perf_counter() - t0
    print(f'[{a.case}] Np={len(P["m"])}, 网格节点={grid["nn"]}, dt={par["dt"]:.3e}, nstep={nstep}, '
          f'夹杂数={len(incl)}, 形函数={cfg["shape"]}')
    R = run_explicit(P, grid, par, kind=cfg['shape'], monitor=mon)
    print(R['timer'].report(a.case))
    # =================================================================
    # 四、数据整理（保存精度、耗时信息；比较与绘图见 explicit/compare_explicit.py）
    # =================================================================
    os.makedirs('results/explicit', exist_ok=True)
    tag = a.tag or f'{a.case}_{cfg["shape"]}_h{a.h}_ppc{a.ppc}_n{nstep}'
    np.savez(f'results/explicit/{tag}.npz', x=R['x'], v=R['v'], F=R['F'], s=R['s'], X0=P['x'],
             phase=P['phase'], **{'h_' + k: v for k, v in R['hist'].items()})
    info = dict(cfg=cfg, Np=R['Np'], nn=grid['nn'], nn_active=R['nn_active'], dt=par['dt'], nstep=nstep,
                t_setup=t_setup, timing=R['timer'].t, t_total=R['timer'].total(),
                t_per_step=R['timer'].total() / max(nstep, 1))
    json.dump(info, open(f'results/explicit/{tag}.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'每步耗时 {info["t_per_step"]*1e3:.2f} ms，结果 -> results/explicit/{tag}.*')


if __name__ == '__main__':
    main()
