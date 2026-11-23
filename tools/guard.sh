#!/usr/bin/env bash
# 计算守卫：所有计时/长计算均通过本脚本启动。
#   用法: tools/guard.sh <命令...>        |  tools/guard.sh --left （查看距截止秒数）
#   - 强制单线程（研究要求：不得使用多线程加速）
#   - 单次计算最长 GUARD_MAX 秒（默认 540）
#   - 可选硬截止：GUARD_DEADLINE=ISO-8601 时间（如 2026-11-24T01:30:00Z）；
#     设置后距截止 < 3 分钟拒绝启动新操作（退出码 99），计算最多运行到截止前 1 分钟。未设置则无截止。
#   （首次夜间执行使用的截止为 2026-11-10T01:30:00Z，已结束。）
MAX=${GUARD_MAX:-540}
if [ -n "$GUARD_DEADLINE" ]; then
  LEFT=$(( $(date -u -d "$GUARD_DEADLINE" +%s) - $(date -u +%s) ))
  if [ "$1" = "--left" ]; then echo "$LEFT"; exit 0; fi
  if [ "$LEFT" -lt 180 ]; then
    echo "[guard] 距硬截止仅剩 ${LEFT}s，拒绝启动新操作。请写入检查点并暂停。" >&2
    exit 99
  fi
  LIMIT=$((LEFT - 60)); [ "$LIMIT" -gt "$MAX" ] && LIMIT=$MAX
else
  if [ "$1" = "--left" ]; then echo "无截止"; exit 0; fi
  LIMIT=$MAX
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=10 "$LIMIT" "$@"
