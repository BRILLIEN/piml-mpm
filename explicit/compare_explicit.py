# -*- coding: utf-8 -*-
"""显式路线比较程序（独立，可选执行）：误差指标与加速比。
用法：python3 -m explicit.compare_explicit <tag> [--plot]
误差指标（相对细尺度基准）：
  e_tip   : 自由端监测点位移时程 L2 相对误差（对共同时间点插值）
  e_tipmax: 最大挠度相对误差
  e_E     : 应变能时程 L2 相对误差
  dE_end  : 末时刻总能量漂移 / 最大应变能
  e_xfin  : 末时刻全体粒子位置 RMS 误差 / 最大挠度
"""
import sys, json
import numpy as np


def load(tag, name):
    d = np.load(f'results/explicit/{tag}_{name}.npz'); return {k: d[k] for k in d.files}


def metrics(ref, r):
    t = ref['h_t']; tr = r['h_t']
    def it(y): return np.interp(t, tr, y)
    # h_mon 列：[x_tip, y_tip, x_root, y_root, x_mid, y_mid, v..., s...]（监测点顺序：自由端、根部、中部）
    uy_ref = ref['h_mon'][:, 1] - ref['h_mon'][0, 1]
    uy = it(r['h_mon'][:, 1]) - r['h_mon'][0, 1]
    ux_ref = ref['h_mon'][:, 0] - ref['h_mon'][0, 0]; ux = it(r['h_mon'][:, 0]) - r['h_mon'][0, 0]
    e_tip = np.sqrt(np.sum((uy - uy_ref) ** 2 + (ux - ux_ref) ** 2) / np.sum(uy_ref ** 2 + ux_ref ** 2))
    e_max = abs(uy.min() - uy_ref.min()) / abs(uy_ref.min())
    Es_ref = ref['h_Es']; Es = it(r['h_Es'])
    e_E = np.sqrt(np.sum((Es - Es_ref) ** 2) / np.sum(Es_ref ** 2))
    Et = r['h_Ek'] + r['h_Es'] + r['h_Wg']
    dE = Et[-1] / np.max(r['h_Es'])
    dEref = (ref['h_Ek'] + ref['h_Es'] + ref['h_Wg'])[-1] / np.max(Es_ref)
    e_x = np.sqrt(np.mean(np.sum((r['x'] - ref['x']) ** 2, 1))) / abs(uy_ref.min())
    return dict(e_tip=e_tip, e_tipmax=e_max, e_E=e_E, dE_end=dE, dE_ref=dEref, e_xfin=e_x)


if __name__ == '__main__':
    tag = sys.argv[1]
    info = json.load(open(f'results/explicit/{tag}.json'))
    ref = load(tag, 'fine'); tf = info['runs']['fine']['t_total']
    print(f'{"方法":<22s}{"nstep":>7s}{"总耗时s":>9s}{"加速比":>8s}{"e_tip":>9s}{"e_tipmax":>9s}{"e_E":>9s}{"ΔE_end":>9s}{"e_xfin":>9s}')
    rows = {}
    for name, ri in info['runs'].items():
        if 'failed' in ri:
            print(f'{name:<22s} 失败: {ri["failed"]}'); continue
        r = load(tag, name)
        mt = metrics(ref, r) if name != 'fine' else dict(e_tip=0, e_tipmax=0, e_E=0, dE_end=(ref['h_Ek'] + ref['h_Es'] + ref['h_Wg'])[-1] / ref['h_Es'].max(), e_xfin=0)
        rows[name] = dict(mt, t=ri['t_total'], speedup=tf / ri['t_total'])
        print(f'{name:<22s}{ri["nstep"]:>7d}{ri["t_total"]:>9.2f}{tf/ri["t_total"]:>8.2f}{mt["e_tip"]:>9.4f}{mt["e_tipmax"]:>9.4f}'
              f'{mt["e_E"]:>9.4f}{mt["dE_end"]:>+9.4f}{mt["e_xfin"]:>9.4f}')
    json.dump(rows, open(f'results/explicit/{tag}_metrics.json', 'w'), indent=1, default=float)
    if '--plot' in sys.argv:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name in rows:
            r = load(tag, name)
            ax[0].plot(r['h_t'], r['h_mon'][:, 1] - r['h_mon'][0, 1], label=name)
            ax[1].plot(r['h_t'], r['h_Ek'] + r['h_Es'] + r['h_Wg'], label=name)
        ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('tip u_y [m]'); ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('E_total [J]')
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'results/explicit/{tag}.png', dpi=120)
