# -*- coding: utf-8 -*-
"""按"目标模块 / 其他模块 / 完整分析"三层口径重算隐式加速比（修订方案的统一口径）。
目标模块 M（隐式）：Newton 线性化与求解核心 —— 产生每次 Newton 修正量所需的全部计算：
  原始 MPM：pattern（稀疏结构）+ tangent（一致切线组装）+ solve（全局线性求解）
  加速方案：basis（多尺度局部算子）+ agg（积分聚合）+ pattern + tangent + project（ΦᵀKΦ）+ solve
其他模块 O：shape、const（本构）、resid（残量）、update（粒子更新）。
三个加速比：S_NL = M_原始 / M_非学习；S_ML = M_非学习 / M_PIML（同一目标模块上的额外加速）；S_tot = 完整_原始 / 完整_PIML。
用法：python3 tools/module_speedup.py results/implicit/ired40_h0025.json"""
import json, sys
M_KEYS = ['basis', 'agg', 'pattern', 'tangent', 'project', 'solve', 'precond', 'precond_basis', 'precond_coarse', 'precond_ilu']


def split(t):
    m = sum(v for k, v in t.items() if k in M_KEYS); return m, sum(t.values()) - m


for f in sys.argv[1:]:
    d = json.load(open(f))['runs']
    base = 'direct'; nl = next((k for k in d if k.startswith('red_') and 'piml' not in k), None)
    ml = next((k for k in d if k.startswith('red_') and 'piml' in k), None)
    Mb, Ob = split(d[base]['timing'])
    print(f'== {f}\n  原始 MPM：目标模块 {Mb:.1f}s（{100*Mb/(Mb+Ob):.1f}%），其他 {Ob:.1f}s，完整 {Mb+Ob:.1f}s')
    for k in (nl, ml):
        if k is None: continue
        M, O = split(d[k]['timing'])
        print(f'  {k:18s}：目标模块 {M:.1f}s，其他 {O:.1f}s，完整 {M+O:.1f}s，模块加速 {Mb/M:.2f}×，完整加速 {(Mb+Ob)/(M+O):.2f}×，'
              f'误差 {d[k].get("tip_err_vs_first", float("nan")):.4f}')
        print('     模块内构成：' + '  '.join(f'{a}:{100*b/M:.0f}%' for a, b in d[k]['timing'].items() if a in M_KEYS))
    if nl and ml:
        Mn, _ = split(d[nl]['timing']); Mm, _ = split(d[ml]['timing'])
        print(f'  ⇒ S_NL（模块）= {Mb/Mn:.2f}×；S_ML（同模块额外）= {Mn/Mm:.2f}×；S_tot（完整）= {sum(d[base]["timing"].values())/sum(d[ml]["timing"].values()):.2f}×')
