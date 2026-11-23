#!/usr/bin/env bash
cd /home/user/piml-mpm
for N in "16 3 3" "32 3 3"; do
  tag=sc3d_$(echo $N|tr ' ' x)
  GUARD_MAX=12000 tools/guard.sh python3 -m implicit3d.case1_TOP3D_Heterogeneous --Nel $N --h 0.125 --dt 0.025 --steps 2 --modes fine_pcg fine_pcg_inexact sub_current subpc3 --subpc_tol 1e-3 --tag $tag > results/logs/$tag.log 2>&1
  echo "$tag exit=$?" >> results/logs/scale3d.log
done
echo DONE >> results/logs/scale3d.log
