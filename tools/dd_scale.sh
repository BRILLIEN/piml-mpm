#!/usr/bin/env bash
cd /home/user/piml-mpm
GUARD_MAX=20000 tools/guard.sh python3 -m implicit.main_implicit --h 0.0025 --nload 10 --n 8 --solvers direct dd_tol2 dd_piml_tol2 --quiet --tag dd_h0025 > results/logs/dd_h0025.log 2>&1
for s in direct dd_tol2 dd_piml_tol2; do
  GUARD_MAX=30000 tools/guard.sh python3 -m implicit.main_implicit --h 0.00125 --nload 10 --n 8 --stop_after 2 --solvers $s --quiet --tag dd_h00125_$s > results/logs/dd_h00125_$s.log 2>&1
done
echo DONE > results/logs/dd_scale.done
