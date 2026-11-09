# -*- coding: utf-8 -*-
"""cell-crossing 动力学批量分析（独立、可选执行）。
指标：
  失败步     ：J<=0 或越界时的步号（无则 —）
  ΔE/Es,max  ：t=0.8 s 时总能量漂移 / 最大应变能（能量守恒性）
  偏移敏感度 ：同一格式/形函数在 3 种网格偏移下自由端位移时程的最大差 / 最大挠度（网格位置依赖性 = 跨格误差的宏观表现）
  根部应力噪声：根部监测粒子 σ_xx 时程二阶差分 RMS / 信号 RMS（高频虚假振荡）
  每步耗时   ：ms
用法：python3 -m explicit.analyze_cc [--plot]
"""
import json, os, re, sys
import numpy as np

SHIFTS = ['0_0', '0.5_0.5', '0.25_0.75']


def load(tag):
    f = f'results/explicit/{tag}.npz'
    if not os.path.exists(f):
        log = open(f'results/logs/{tag}.log').read() if os.path.exists(f'results/logs/{tag}.log') else ''
        m = re.search(r'step (\d+)', log.split('Error')[-1]) if 'Error' in log else None
        return None, (m.group(1) if m else '?')
    d = np.load(f); info = json.load(open(f'results/explicit/{tag}.json'))
    return dict(d=d, info=info), None


rows = []; out = {}
for sch in ['USL', 'MUSL']:
    for shape in ['linear', 'qbspline']:
        tips = []; res = dict()
        for sh in SHIFTS:
            r, fail = load(f'cc_{sch}_{shape}_{sh}')
            if r is None:
                res[sh] = dict(fail=fail); continue
            d = r['d']; t = d['h_t']; mon = d['h_mon']
            E = d['h_Ek'] + d['h_Es'] + d['h_Wg']; Es = d['h_Es']
            uy = mon[:, 1] - mon[0, 1]; tips.append((t, uy))
            s11 = mon[:, 12 + 1]                                   # 列：x,y×3 | v×6 | s11×3 -> 根部 s11
            hf = float(np.sqrt(np.mean(np.diff(s11, 2) ** 2)) / np.sqrt(np.mean(s11 ** 2)))
            res[sh] = dict(dE=float(E[-1] / Es.max()), hf=hf, ms=1e3 * r['info']['t_per_step'], tipmin=float(uy.min()))
        sens = None
        if len(tips) >= 2:
            tt = tips[0][0]
            Y = np.array([np.interp(tt, t, u) for t, u in tips])
            sens = float(np.max(np.abs(Y - Y.mean(0))) / np.abs(Y).max())
        out[f'{sch}_{shape}'] = dict(res=res, shift_sensitivity=sens)
        for sh in SHIFTS:
            q = res[sh]
            if 'fail' in q:
                print(f'{sch:5s} {shape:9s} shift={sh:10s} 失败（step {q["fail"]}）')
            else:
                print(f'{sch:5s} {shape:9s} shift={sh:10s} ΔE/Esmax={q["dE"]:+.4f}  根部σxx噪声={q["hf"]:.2e}  '
                      f'自由端最小 u_y={q["tipmin"]:+.4f}  每步 {q["ms"]:.2f} ms')
        print(f'      -> 网格偏移敏感度（自由端时程最大偏差/最大挠度）= {sens if sens is None else round(sens, 4)}')
json.dump(out, open('results/explicit/cc_analysis.json', 'w'), indent=1)
if '--plot' in sys.argv:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for i, sch in enumerate(['USL', 'MUSL']):
        for shape, ls in [('linear', '--'), ('qbspline', '-')]:
            for sh in SHIFTS:
                r, _ = load(f'cc_{sch}_{shape}_{sh}')
                if r is None: continue
                d = r['d']
                ax[i, 0].plot(d['h_t'], d['h_mon'][:, 1] - d['h_mon'][0, 1], ls, label=f'{shape} {sh}')
                ax[i, 1].plot(d['h_t'], d['h_mon'][:, 13], ls, lw=0.7, label=f'{shape} {sh}')
        ax[i, 0].set_title(f'{sch}: tip u_y'); ax[i, 1].set_title(f'{sch}: root sigma_xx'); ax[i, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig('results/explicit/cc_analysis.png', dpi=110)
