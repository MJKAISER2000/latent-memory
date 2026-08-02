# Latent Twins → LLM context memory: test plan

Reference: Chung, Bu & Verma (2026), *Physics-Conforming Latent Twins*, arXiv:2606.15053v2.

## The claim under test

The paper's headline result (Fig. 5/6) is that a surrogate with the *right conserved
structure* extrapolates past its training horizon, beating a PINN that had access to
the governing equations — with 508 parameters vs. 12,738. Structure beat residual
minimisation out of distribution.

**Transferred hypothesis:** a context memory whose generator exactly conserves a
designated subspace will retain what lives in that subspace far past the sequence
length it was trained on, while the orthogonal complement decays monotonically.

The one modification the paper does not make, and we must: **their flow is autonomous
(`m_{s→t}(z)`, no inputs). Context memory is driven.** Under additive forcing, exact
conservation becomes an exact *balance law*:

```
C^T z_t = C^T z_s + Σ_{k=s}^{t-1} C^T g_k
```

Verified exact to 3.2e-13 (float64) over 128 driven steps in `latent_twin_memory.py`'s
self-test — by construction, with no loss term. This reframes the learning problem:
conservation is free, and all the work goes into making `C^T g_k` *mean* something.
That is the input-side analogue of the paper's pullback matching (their Eq. 7).

## Kill criteria

Stated up front so we don't rationalise a null result:

- **L0 fails** if `pinned` shows no advantage over `dissipative` at 16× extrapolation.
  Then the structured generator buys nothing and levels 1–3 are pointless.
- **L0.5 fails** if conserve-without-anchor performs *as well as* anchored. That would
  contradict the paper's Fig. 10 and mean our setup isn't measuring what we think.
- **L2 fails** if the memory can't beat "just keep the last N tokens" at equal budget.
  This is the honest bar and most compression schemes lose to it.

---

## Level 0 — synthetic ledger (`toy_ledger.py`) ✅ built

Running total with distractors. Train at K=64, test at K=1024. Four generators:
`free` / `dissipative` (Mamba corner) / `conservative` (DeltaNet corner) / `pinned`.

**Expectation, honestly stated:** `pinned` should *not* be error-free. Each block's
flux estimate `C^T g_k` carries error and those errors random-walk, so absolute ledger
error grows ~√t. The claim is √t accumulation vs. compounding drift — not zero error.
If `pinned` looks perfect, suspect a leak (e.g. the content task isn't actually
pressuring the latent).

Fairness notes already applied:
- All modes initialised so `exp(dt·W) ≈ I` (`init_spectral=0.03`). With the naive
  `1/√n_z` init the unconstrained baseline NaNs *before training starts* — a 64-step
  rollout of a spectral-radius-1 generator is `e^64`. That's a strawman, not a result.
- Baselines get an equally expressive **learned** linear readout. The only difference
  between them and `pinned` is whether the generator is constrained to conserve it.
- The content-reconstruction loss is load-bearing. Without it the model dedicates the
  whole latent to the ledger and the comparison is vacuous.

## Level 0.5 — anchor transport (the sharper version of their Fig. 10)

**Do this before touching a real LM.** Their most transferable finding: with hard
skew-symmetry the latent invariant is conserved to machine precision *and tells you
nothing* about the physical Hamiltonian unless the pullback loss ties them together.
Conservation without anchoring is conservation of nothing.

The naive port — train with no ledger supervision, observe the ledger is meaningless —
is true but nearly tautological in our setup, since the ledger loss *is* the anchor.

The sharper experiment inverts it and makes a prediction **unique to the pinned model**:

> Supervise the ledger at **one horizon only** (t = 64). Test at t = 1…1024.

The pinned model should get every other horizon *for free*, because exact conservation
transports the single anchor to all times — one supervised time point pins the whole
trajectory. The baselines have no such mechanism and should need supervision at every
prefix to match.

This is worth more than the vanilla ablation: it's a falsifiable, structure-specific
claim, and if it holds it's the cleanest argument for the whole approach — *structure
converts sparse supervision into dense supervision.* It also maps onto something real:
you rarely have per-token labels for what a memory should be retaining.

If conserve-without-anchor does *fine* in the vanilla ablation, something is wrong with
the experiment — the task is leaking the answer through another path. Bug, not finding.

## Level 1 — real LM hidden states

**Crucial practical move: precompute once, train offline.** Run the frozen LM over a
long-context corpus, dump pooled residual-stream blocks to disk as tensors, then train
the memory module on those. The memory module is tiny (<1M params) — once the hidden
states are cached, the 6 GB VRAM ceiling stops mattering.

- Block the stream at 64 or 128 tokens; `x_k` = mean-pooled residual stream at some
  mid-to-late layer.
- Train enc/dec/generator with recon + a retrieval-style probe objective.
- **Metric:** needle-in-haystack accuracy as a function of needle→query distance,
  comparing pinned vs. unpinned content. The pinned subspace should show a materially
  flatter decay curve.

## Level 2 — KV cache proper

`x_k` = flattened K,V block; decoder reconstructs KV entries consumed by attention.
Only worth doing if L1 shows a flat retention curve.

- **Metric:** perplexity delta and needle retrieval at matched memory footprint.
- **The honest baseline is sliding-window + sink tokens**, not full attention. Most
  learned KV compression loses to it. If we can't beat it, say so.
- Two sub-variants: (a) memory *replaces* distant KV; (b) memory is an *auxiliary*
  channel attended alongside a sliding window. (b) is much more likely to work and is
  the one I'd build first.

### 2b — chunked scan (needed only at this level)

The interaction-picture prefix sum (`y_k = exp(-kΔW) z_k`, giving O(1) random access)
is **numerically unusable** for dissipative `W`: `exp(-kW)` grows without bound. Use
the standard chunked formulation instead — local prefix sums within chunks of length
C, plus chunk-boundary states. Forward binary powering (implemented in `ExpFlow`) is
stable and gives O(log K) *homogeneous* jumps; only the driven random-access path
needs chunking.

## Level 3 — physics in context (the other branch)

Separate track from 1/2. The latent as a context-resident, structurally-audited
scratchpad:

- Train a latent twin on a simulation (scipy `odeint` for ODEs; the paper used Dedalus
  for PDEs, but heat/wave on a periodic box via FFT is ~50 lines and avoids the dep).
- Expose tools to the LLM: `propagate(z, s, t)`, `decode(z)`, `invariants(z)`.
- `invariants(z)` returns numbers the model **cannot corrupt by reasoning about them**.
- **Metric:** invariant drift over long horizons, and hallucination rate on
  "what is the energy at t=X" vs. an LLM given the raw trajectory in context.
- Bonus, cheap and useful: the compatibility defect `|C(d(z)) − C_Z(z)|` computed online
  is an OOD/staleness detector. Spikes mean the memory has drifted off its training
  manifold → signal "low confidence, go back to ground truth."

---

## Hardware reality check

RTX 3060 Laptop, **6 GB VRAM**. This is the binding constraint.

- Fine: 0.5B model in fp16 (~1 GB) with 32k–64k context. Qwen2.5-0.5B's GQA KV cache is
  ~12 KB/token, so 64k tokens ≈ 800 MB. Workable.
- Not fine: anything ≥3B, or 128k-context experiments with a 1.5B+ model.
- Mitigation: the precompute-to-disk strategy in L1. Hidden-state extraction is a
  single forward pass we run once; memory-module training then never touches the LM.
- L0 and L0.5 run entirely on the GPU in minutes and need none of this.
