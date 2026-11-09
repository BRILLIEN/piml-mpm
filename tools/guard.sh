#!/usr/bin/env bash
# 截止时间守卫：所有长时计算均通过本脚本启动。
#   用法: tools/guard.sh <命令...>
#   - 硬截止 2026-11-10 09:30 (UTC+8) = 2026-11-10T01:30:00Z
#   - 距截止 < 3 分钟时拒绝启动新计算（退出码 99）
#   - 单次计算最多运行至截止前 1 分钟，且不超过 GUARD_MAX 秒（默认 540）
#   - 强制单线程（研究要求：不得使用多线程加速）
DEADLINE_EPOCH=$(date -u -d "2026-11-10T01:30:00Z" +%s)
NOW=$(date -u +%s)
LEFT=$((DEADLINE_EPOCH - NOW))
if [ "$1" = "--left" ]; then echo "$LEFT"; exit 0; fi
if [ "$LEFT" -lt 180 ]; then
  echo "[guard] 距硬截止仅剩 ${LEFT}s，拒绝启动新操作。请写入检查点并暂停。" >&2
  exit 99
fi
MAX=${GUARD_MAX:-540}
LIMIT=$((LEFT - 60)); [ "$LIMIT" -gt "$MAX" ] && LIMIT=$MAX
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=10 "$LIMIT" "$@"
