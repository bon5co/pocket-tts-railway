#!/usr/bin/env bash
# Pin every math library to one thread. Containers report the HOST's core count, and torch/OpenMP size their pools
# from it; on an 8 vCPU Railway container that oversubscribes (measured 5x on torch elsewhere in this repo). Pocket
# TTS is a 100M model run at batch 1 and is fastest on ONE thread (measured 2026-11-25: 0.46 real-time factor on one
# thread, 0.96 on two, 1.04 on four), so the cap here is 1, not the cgroup quota. The quota is only logged.
set -euo pipefail
log() { echo "[entrypoint] $*"; }

cpu_quota() {
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max || return 1
        [ "$quota" = "max" ] && return 1
        echo $(( (quota + period - 1) / period )); return 0
    fi
    return 1
}

VISIBLE="$(nproc)"
if CPUS="$(cpu_quota)"; then :; else CPUS="$VISIBLE"; fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log "cpu: visible=$VISIBLE quota=$CPUS -> threads=1 PORT=${PORT:-unset}"
exec "$@"
