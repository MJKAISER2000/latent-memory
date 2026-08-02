#!/usr/bin/env bash
# Environment setup. Works on a laptop and on a SLURM cluster login node.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

echo "== module load (cluster only; harmless if absent) =="
if command -v module >/dev/null 2>&1; then
  module load cuda    2>/dev/null || echo "  (no cuda module)"
  module load python  2>/dev/null || echo "  (no python module)"
fi

echo "== venv =="
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== deps =="
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo "== dirs =="
mkdir -p cache data results logs

echo "== verify =="
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu: {p.name}  {p.total_memory/1e9:.1f} GB")
PY

echo "== structural self-test =="
python latent_twin_memory.py | tail -4

cat <<'EOF'

Ready. Next:
  python toy_ledger.py                                   # L0
  python data/haystack.py --task retrieve --n 200 --out data/retrieve_2k
  python precompute_states.py --jsonl data/retrieve_2k.jsonl --out cache/retrieve_2k.pt

On SLURM, submit instead of running directly:
  sbatch scripts/job.sbatch precompute
EOF
