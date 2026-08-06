# Environment provenance

- captured: 2026-08-06T02:26:59
- commit: 11e3bd31bd1ac931c25bb6aa3d548df85e88a1e9 (main)
- dirty: True
- os: Windows-10-10.0.26200-SP0
- python: 3.11.9
- torch: 2.5.1+cu121
- cuda available: True (build 12.1)
- gpu: NVIDIA GeForce RTX 3060 Laptop GPU
- cudnn: 90100

Determinism note: experiments seed `torch.manual_seed` (weights) and
per-device `torch.Generator` objects (data), so the data stream is
device-dependent (CPU vs CUDA generators differ) and results are
expected to reproduce on the same device class only. cuDNN/cuBLAS
nondeterminism is not explicitly disabled; reruns on the recorded GPU
have matched logged values to the printed 3-decimal precision.
