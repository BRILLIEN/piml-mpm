#!/usr/bin/env bash
# 阶段A：原始 MPM 与加速方案在大规模下的目标模块计时（单线程，每个求解器独立进程以分别记录峰值内存）
cd /home/user/piml-mpm
run() { # h stop_after solver tag
  GUARD_MAX=30000 tools/guard.sh python3 -m implicit.main_implicit --h $1 --nload 40 --n 8 --stop_after $2 --solvers $3 --quiet --tag $4 > results/logs/$4.log 2>&1
  echo "$4 exit=$?" >> results/logs/phaseA_scale.log
}
run 0.00125 2 direct           pA_h00125_direct
run 0.00125 2 red_os_piml_agg  pA_h00125_red_piml
run 0.00125 2 red_os_agg       pA_h00125_red_nl
run 0.000625 1 red_os_piml_agg pA_h000625_red_piml
run 0.000625 1 direct          pA_h000625_direct
echo DONE >> results/logs/phaseA_scale.log
