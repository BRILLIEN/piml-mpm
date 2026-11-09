#!/usr/bin/env bash
# h=0.0025 区间物理时长检验：K=150（≈4.8 ms），复用 t10 细尺度基准
cd /home/user/piml-mpm
until grep -q DONE results/logs/t13.log 2>/dev/null; do sleep 30; done
GUARD_MAX=9000 tools/guard.sh python3 -m explicit.main_explicit_ms --h 0.0025 --tend 0.4 --n 8 --K 150 --agg 1 0 --basis os_piml os --theta 0.1 --skip_fine --edge_suffix _c100 --out_every 80 --tag t14 > results/logs/t14.log 2>&1
python3 - >> results/logs/t14.log 2>&1 <<'PY'
import json, shutil
a=json.load(open('results/explicit/t10.json')); b=json.load(open('results/explicit/t14.json'))
b['runs']={'fine':a['runs']['fine'],**b['runs']}; json.dump(b,open('results/explicit/t14.json','w'))
shutil.copy('results/explicit/t10_fine.npz','results/explicit/t14_fine.npz')
PY
python3 -m explicit.compare_explicit t14 >> results/logs/t14.log 2>&1
echo DONE >> results/logs/t14.log
