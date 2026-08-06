#!/usr/bin/env bash
#SBATCH --job-name=latent-twin
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# TEMPLATE -- adjust partition/account/constraint for your cluster before use:
#   #SBATCH --partition=gpu
#   #SBATCH --account=<your-alloc>
#   #SBATCH --constraint=a100
#
# Usage:  sbatch scripts/job.sbatch <stage>
#   stages: selftest | l0 | seeds | data | precompute
#           | frontier | mechanism | snr | stats | forget   (live-claim track)
#
# The whole point of precompute is that it runs ONCE on the cluster and the
# resulting cache/*.pt transfers back to any machine. Memory-module training
# afterwards is small enough to run on the laptop.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null || true
fi
# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate

mkdir -p cache data results logs
STAGE="${1:-selftest}"
echo "=== stage=$STAGE  node=$(hostname)  $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$STAGE" in
  selftest)
    python latent_twin_memory.py
    ;;

  l0)
    python -u toy_ledger.py | tee "results/L0_${SLURM_JOB_ID:-local}.txt"
    ;;

  seeds)
    python -u seed_check.py | tee "results/L0_seeds_${SLURM_JOB_ID:-local}.txt"
    ;;

  data)
    for task in retrieve update count; do
      python data/haystack.py --task "$task" --n 500 --n-filler 2000 \
        --out "data/${task}_2k"
    done
    ;;

  precompute)
    # Bigger model + longer context than the 6 GB laptop allows.
    for task in retrieve update count; do
      python -u precompute_states.py \
        --jsonl "data/${task}_2k.jsonl" \
        --out   "cache/${task}_2k.pt" \
        --model Qwen/Qwen2.5-1.5B \
        --chunk 4096 \
        --context-mode local
    done
    ;;

  frontier)
    python -u l0_frontier.py | tee "results/L0_frontier_${SLURM_JOB_ID:-local}.txt"
    ;;

  mechanism)
    python -u l0_mechanism.py | tee "results/L0_mechanism_${SLURM_JOB_ID:-local}.txt"
    ;;

  snr)
    python -u l0_snr.py | tee "results/L0_snr_${SLURM_JOB_ID:-local}.txt"
    ;;

  stats)
    python -u l0_stats.py | tee "results/L0_stats_${SLURM_JOB_ID:-local}.txt"
    ;;

  forget)
    python -u l0_forget.py | tee "results/L0_forget_${SLURM_JOB_ID:-local}.txt"
    ;;

  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac

echo "=== done $(date -Is) ==="
