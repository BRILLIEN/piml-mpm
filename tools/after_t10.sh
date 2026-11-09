#!/usr/bin/env bash
# t10 完成后：h=0.0025 粒子积分聚合 agg=1 对比（复用 t10 细尺度基准）
cd /home/user/piml-mpm
until grep -q DONE results/logs/t10.log 2>/dev/null; do sleep 30; done
GUARD_MAX=6000 tools/guard.sh python3 -m explicit.main_explicit_ms --h 0.0025 --tend 0.4 --n 8 --K 50 --agg 1 --basis os_piml --theta 0.1 --skip_fine --edge_suffix _c100 --out_every 80 --tag t13 > results/logs/t13.log 2>&1
python3 - >> results/logs/t13.log 2>&1 <<'PY'
import json, shutil
a=json.load(open('results/explicit/t10.json')); b=json.load(open('results/explicit/t13.json'))
b['runs']={**a['runs'],**b['runs']}; json.dump(b,open('results/explicit/t13.json','w'))
for nm in a['runs']: shutil.copy(f'results/explicit/t10_{nm}.npz',f'results/explicit/t13_{nm}.npz')
PY
python3 -m explicit.compare_explicit t13 >> results/logs/t13.log 2>&1
echo DONE >> results/logs/t13.log
