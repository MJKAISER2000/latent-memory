# Latent Twins → LLM context memory

Transplanting the structured-generator machinery from Chung, Bu & Verma (2026),
*Physics-Conforming Latent Twins* (arXiv:2606.15053v2), into a driven context memory.

**Core idea.** Build the latent generator so that a *designated subspace* is exactly
conserved while everything orthogonal to it decays monotonically:

```
P = I − C Cᵀ                     # projector onto the complement of pinned dirs C
K = P (A − Aᵀ) P                 # skew  → norm-preserving rotation on C-perp
S = −(PB)(PB)ᵀ − eps·P           # psd   → monotone decay on C-perp
W = K + S                        # ⟹ CᵀW = 0  and  W + Wᵀ ⪯ 0
```

Under additive forcing (`z_{k+1} = exp(dt·W) z_k + g_k`) exact conservation becomes an
exact **balance law**, verified to 3.2e-13 over 128 driven steps:

```
Cᵀz_t = Cᵀz_s + Σ Cᵀg_k          # the ledger changes only by what came in the door
```

Conservation is then *free*, and the whole learning problem becomes making `Cᵀg_k` mean
something — the input-side analogue of the paper's pullback matching (their Eq. 7).

---

## Setup

### Windows (local dev)

```bash
pip install -r requirements.txt
```

### HPC / Linux

```bash
bash scripts/setup.sh
```

Verify the structural guarantees hold on your install (should print defects ~1e-7 in
float32, ~1e-16 in float64):

```bash
python latent_twin_memory.py
```

---

## Pipeline

```
data/haystack.py       synthesize corpora  →  .jsonl (text + durable spans)
precompute_states.py   frozen LM forward   →  .pt    (block-pooled states + labels)
[next] train_memory.py train memory module →  retention curves
```

---

## Command log

Everything run so far, in order. Reproduces the current state from scratch.

### 0. Structural self-test

```bash
python latent_twin_memory.py
```

Verifies `max|CᵀW|`, skewness of `K`, `eig(W+Wᵀ) ≤ 0`, ledger drift over t=10⁴,
binary-powering vs. `torch.matrix_exp`, and the driven balance law. All pass.
Residuals are float32 precision — confirmed by rerunning under `torch.float64`.

### 1. L0 — synthetic ledger, extrapolation test

```bash
python toy_ledger.py
```

Trains at K=64, tests at K=1024, across four generator modes. ~20 min
(`free` alone takes ~15 min — Padé scaling-and-squaring slows as ‖W‖ grows).
Output: [results_L0_seed0.txt](results_L0_seed0.txt).

| mode | t=64 | t=128 | t=256 | t=512 | t=1024 |
|---|---|---|---|---|---|
| free | 0.0469 | 0.0370 | 0.0384 | 0.0524 | 0.0902 |
| dissipative *(Mamba corner)* | 0.0435 | 0.0356 | 0.0790 | 0.1771 | 0.3397 |
| conservative *(DeltaNet corner)* | 0.0470 | 0.0384 | 0.0582 | 0.1597 | 0.5021 |
| **pinned** | 0.0456 | 0.0359 | **0.0229** | **0.0148** | **0.0116** |

Tied in-distribution; `pinned` is 29×/43× better than the baselines at 16×
extrapolation. Its relative error falls as `1/√t` — predicted 0.250, observed 0.254.

Ignore the t=16 column: `M_16 ≈ 0`, so relative error explodes on the denominator.

### 2. Seed robustness

```bash
python -u seed_check.py | tee results_L0_seeds.txt
```

3 seeds × 3 modes (`free` excluded — 10× slower). ~15 min.

### 3. Synthesize haystack data

```bash
python data/haystack.py --task retrieve --n 200 --n-filler 2000 --out data/retrieve_2k
python data/haystack.py --task update   --n 200 --n-filler 2000 --out data/update_2k
python data/haystack.py --task count    --n 200 --n-filler 2000 --out data/count_2k
```

`--n-filler 2000` ≈ 7.4k tokens/sample. Add `--tokenizer Qwen/Qwen2.5-0.5B` to also
report block alignment. Sweep `--depths` to vary needle→query distance.

### 4. Precompute LM states

```bash
python precompute_states.py --jsonl data/retrieve_2k.jsonl --out cache/retrieve_2k.pt
```

Defaults: Qwen2.5-0.5B, penultimate layer, 64-token blocks, `--context-mode local`.
Lower `--chunk` if you OOM on 6 GB.

---

## Design decisions worth not forgetting

**`--context-mode local` is the honest default.** Each chunk is encoded independently,
so block representations see only their own chunk. We are testing whether the *memory
module* carries long-range information; letting the LM attend over the full prefix lets
the LM do the retention instead and confounds the measurement. `full` exists as a
topline reference only. **Report which mode you used — they are not comparable.**

**Distractors mimic the task's own template.** Every durable sentence has surface-form
twins with wrong entities scattered through the filler. Without this, `count` and
`update` collapse to template matching and the retention curve measures nothing.
(Verified: 63 deposit-form lines per sample, 8 belong to the target entity.)

**Init all generator modes near-identity** (`init_spectral=0.03`). With the natural
`1/√n_z` init the unconstrained baseline has spectral radius ~1, so a 64-step rollout is
`exp(W)^64 ≈ e^64` and it NaNs *before training starts*. Any comparison against
"unconstrained" without this fix is a strawman.

**The O(1) random-access trick does not survive dissipation.** The interaction-picture
prefix sum (`y_k = exp(−kΔW)z_k`) needs `exp(−kW)`, which blows up exactly when `W` is
dissipative. Forward binary powering is stable and gives O(log K) *homogeneous* jumps;
the driven random-access path must be chunked. Bites at L2, not before.

---

## Prior work

Much of the machinery is well-trodden; the application and one or two pieces are not.

**Structured recurrent generators — established.**
- [Lipschitz RNN](https://arxiv.org/abs/2006.12070) (Erichson et al., ICLR 2021) is the
  closest on the construction: an explicit *symmetric-skew decomposition* of the
  hidden-to-hidden matrix. Essentially our `K + S`, motivated by gradient stability
  rather than by pinning a semantic subspace.
- [AntisymmetricRNN](https://www.semanticscholar.org/paper/e2c8a6b49cd999b16ac4dcfdc375563a6932b1c7)
  (Chang et al., 2019) — skew-symmetric recurrence for stable dynamics.
- expRNN / unitary RNNs — `exp(skew)` parameterization, our `conservative` mode.
- [Mamba](https://arxiv.org/pdf/2312.00752) / S4 / HiPPO — diagonal negative-real `A`,
  our `dissipative` mode. [StableSSM](https://arxiv.org/pdf/2311.14495) on
  reparameterization for stable memory.

**Latent memory for long context — active and crowded.**
- [IndexMem](https://arxiv.org/html/2605.25475) (May 2026) is the most directly
  competitive: a 0.52M-param latent memory that compresses evicted KV and supplies
  residual readouts; +25 pts on RULER under aggressive eviction, stable NIAH.
  **Its state update is `M ← λM + η Σ Linear(k)⊗vᵀ` — scalar exponential decay, no
  structural constraint.** That is precisely the corner that drifts worst in our L0.
- [KVzip](https://arxiv.org/pdf/2505.23416), NestedKV, KV-CAR, Neural KV compaction.

**What I could not find prior work on** (caveat: absence of evidence — this literature
is enormous and fast-moving):
- The `CᵀW = 0` projection used to pin a *semantically supervised* subspace in an LLM
  memory, as opposed to conserving norm or decaying uniformly.
- The **balance-law / auditable-ledger** framing for driven memory.
- The **anchor-transport** prediction (supervise one horizon, get all horizons free).

Adjacent but distinct: [Symmetry-Protected Lyapunov Neutral Modes](https://arxiv.org/abs/2605.03338)
(2026) derives zero-Lyapunov directions from *group equivariance* — theoretical, not
applied to LMs. Same destination, different mechanism.

---

## Status

- [x] Structured generator + verified guarantees
- [x] L0 synthetic ledger — passed kill criterion
- [x] Seed robustness check
- [x] Haystack synthesizer (retrieve / update / count)
- [x] LM state precompute
- [ ] L0.5 anchor transport
- [ ] L1 memory training on real states + retention curves
- [ ] L2 KV cache, vs. sliding-window + IndexMem-style scalar-decay baseline

See [PLAN.md](PLAN.md) for the full ladder and kill criteria.
