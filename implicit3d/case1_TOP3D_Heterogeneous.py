# -*- coding: utf-8 -*-
"""
三维双端固支异质梁（Chen & Li 2024, CMAME, §8.1.2 图 20）—— 隐式 MPM 主脚本（Python 移植版，单线程）。
结构：基础条件设置 → 算例设置 → 求解过程（各方案独立从同一初态推进，逐步轮换执行顺序）→ 数据整理与比较。
运行：tools/guard.sh python3 -m implicit3d.case1_TOP3D_Heterogeneous --Nel 32 3 3 --h 0.125 --dt 0.025 --steps 6 --modes fine_pcg sub_current subpc
"""
import argparse, copy, json, os, time
import numpy as np
from implicit3d.beam_kernels import setup_heterogeneous_beam
from implicit3d.implicit_step_heterogeneous import implicit_step_heterogeneous

ROOT = os.path.dirname(os.path.abspath(__file__))


def errors(sa, sb, grid):
    """相对误差（以 b 为参考）：位移、速度、Cauchy 应力（Frobenius）、变形梯度增量、分块应变能。"""
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    out = dict(disp=rel(sa['x'] - sa['x0'], sb['x'] - sb['x0']), vel=rel(sa['v'], sb['v']),
               stress=rel(sa['sig'], sb['sig']),
               F=float(np.linalg.norm(sa['F'] - sb['F']) / max(np.linalg.norm(sb['F'] - np.eye(3)[:, :, None]), 1e-300)))
    soft = grid['phase'] > 0
    out['stress_soft'] = rel(sa['sig'][:, soft], sb['sig'][:, soft]); out['stress_matrix'] = rel(sa['sig'][:, ~soft], sb['sig'][:, ~soft])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Nel', type=int, nargs=3, default=[32, 3, 3])
    ap.add_argument('--fine', type=int, default=5)
    ap.add_argument('--h', type=float, default=0.125)
    ap.add_argument('--dt', type=float, default=0.025)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--modes', nargs='+', default=['fine_pcg', 'sub_current', 'subpc'])
    ap.add_argument('--omega', type=float, default=1.0)
    ap.add_argument('--subpc_tol', type=float, default=3e-10, help='子结构预条件细空间 PCG 的线性相对容差（非精确 Newton）')
    ap.add_argument('--looseTol', type=float, default=1e-4, help='名称以 _loose 结尾的方案所用 Newton 相对容差（完整细空间残余力）')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    # ============ 一、基础条件设置 ============
    Nelx, Nely, Nelz = a.Nel; fine = a.fine
    p = dict(cells=[Nelx * fine, Nely * fine, Nelz * fine], fine=fine, h=a.h, ppc=2, pad=[5, 5, 5],
             E=1e5, Einclusion=1e2, nu=0.30, rho=10.0, gravity=np.array([0.0, -25.0, 0.0]),
             dt=a.dt, steps=a.steps, newtonTol=1e-8, absTol=1e-9, activeTol=1e-12, omega=a.omega,
             subpc_tol=a.subpc_tol, looseTol=a.looseTol)
    print(f'本构：可压缩 Neo-Hookean（E={p["E"]:.0e}，软夹杂 E={p["Einclusion"]:.0e}，ν={p["nu"]}），ρ={p["rho"]}，重力 {p["gravity"][1]}')
    print(f'求解：隐式平均加速度 Newmark，Δt={p["dt"]}，{p["steps"]} 步；逐轮一致切线；Newton 相对容差 {p["newtonTol"]}；'
          f'PCG 相对容差 3e-10；子结构 {Nelx}×{Nely}×{Nelz}，每块 {fine}³ 细单元，细格 h={p["h"]}；单线程')
    # ============ 二、算例设置 ============
    s0, grid, inclusions = setup_heterogeneous_beam(p, os.path.join(ROOT, 'data'))
    s0['x0'] = s0['x'].copy()
    print(f'粒子 {s0["x"].shape[1]}（软夹杂内 {int(np.sum(grid["phase"] > 0))}），背景节点 {grid["node"].shape[0]}')
    # ============ 三、求解过程 ============
    S = {m: copy.deepcopy(s0) for m in a.modes}
    rows = {m: [] for m in a.modes}
    caches = {m: {} for m in a.modes}                      # 各方案独立的跨步缓存（切线复用）
    errs = {m: [] for m in a.modes if m != 'fine_pcg'}
    show_steps = set(np.unique(np.round(np.linspace(1, p['steps'], min(5, p['steps']))).astype(int)).tolist())
    for step in range(1, p['steps'] + 1):
        order = a.modes[(step - 1) % len(a.modes):] + a.modes[:(step - 1) % len(a.modes)]   # 逐步轮换执行顺序
        for m in order:
            S[m], caches[m], row = implicit_step_heterogeneous(S[m], caches[m], grid, p, m, step, step in show_steps)
            rows[m].append({k: v for k, v in row.items() if k != 'res'} | dict(res=[list(map(float, q)) for q in row['res']]))
        if 'fine_pcg' in S:
            for m in errs:
                e = errors(S[m], S['fine_pcg'], grid); errs[m].append(dict(step=step, **e))
                print(f'  [误差 vs 细解] {m} 步 {step}: 位移 {e["disp"]:.3e} 速度 {e["vel"]:.3e} 应力 {e["stress"]:.3e} '
                      f'（基体 {e["stress_matrix"]:.3e}，软夹杂 {e["stress_soft"]:.3e}）变形梯度增量 {e["F"]:.3e}')
    # ============ 四、数据整理 ============
    for m in a.modes:
        tot = sum(r['total'] for r in rows[m])
        print(f'[{m}] 总步数 {p["steps"]}，总耗时 {tot:.2f} s，步均 {tot / p["steps"]:.2f} s')
    os.makedirs('results/implicit3d', exist_ok=True)
    tag = a.tag or f'beam3d_N{Nelx}x{Nely}x{Nelz}_h{a.h}_dt{a.dt}'
    json.dump(dict(p={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}, rows=rows, errors=errs),
              open(f'results/implicit3d/{tag}.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('保存 ->', f'results/implicit3d/{tag}.json')


if __name__ == '__main__':
    main()
