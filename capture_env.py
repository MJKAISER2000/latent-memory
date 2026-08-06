"""
Environment provenance capture -- audit fix.

None of the results/*.txt logs record the environment that produced them
(torch/CUDA versions, GPU, driver, commit). This writes results/ENV.md with the
current environment and git state so future reproductions have a reference
point, and can be re-run before any experiment batch.

Additive only: no existing script or log format is modified.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<{e}>"


def main():
    import torch

    gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    lines = [
        "# Environment provenance",
        "",
        f"- captured: {datetime.now().isoformat(timespec='seconds')}",
        f"- commit: {sh('git rev-parse HEAD')} ({sh('git rev-parse --abbrev-ref HEAD')})",
        f"- dirty: {bool(sh('git status --porcelain'))}",
        f"- os: {platform.platform()}",
        f"- python: {platform.python_version()}",
        f"- torch: {torch.__version__}",
        f"- cuda available: {torch.cuda.is_available()} "
        f"(build {torch.version.cuda})",
        f"- gpu: {gpu}",
        f"- cudnn: {torch.backends.cudnn.version()}",
        "",
        "Determinism note: experiments seed `torch.manual_seed` (weights) and",
        "per-device `torch.Generator` objects (data), so the data stream is",
        "device-dependent (CPU vs CUDA generators differ) and results are",
        "expected to reproduce on the same device class only. cuDNN/cuBLAS",
        "nondeterminism is not explicitly disabled; reruns on the recorded GPU",
        "have matched logged values to the printed 3-decimal precision.",
    ]
    out = Path(__file__).parent / "results" / "ENV.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[2:11]))


if __name__ == "__main__":
    main()
