# latent-memory: the idea, what I found, and what's wrong with it

One page, honest version. Every number here gets recomputed from the raw logs by a
script (`results/VERIFICATION.md`, 64/64 checks passing), because I burned myself
early on trusting numbers I'd typed by hand. Everything ran on my laptop's RTX 3060.

## The idea

I was reading a scientific-ML paper (Chung, Bu & Verma, *Physics-Conforming Latent
Twins*) where they build simulators whose latent dynamics *can't* violate
conservation laws — not "trained not to," literally can't, it's baked into the
matrix algebra. And the thought was: LLMs are terrible at holding onto facts over
long contexts. What if you built a memory the same way physicists build these
simulators, where the important stuff is mathematically impossible to forget?

Concretely: a small recurrent state `z` that updates as text streams by, with a
transition matrix built so one designated subspace is exactly conserved
(`CᵀW = 0`) while everything else can decay. When you drive it with inputs you get
what I've been calling a balance law — the protected "ledger" part of the memory
changes *only* by explicit deposits, and I can verify that to about 1e-13. So the
question became: does a memory with hard guarantees actually beat one that just
learns to remember?

## What I found

**1. Memories with learnable forgetting rates fail long-horizon retention, and
it's training's fault.** Train one on 64-step sequences at normal settings and the
forgetting rate converges to ~8e-4 — which I eventually figured out is the actual
*optimum* of the training loss (I measured the loss landscape; there's a real
interior minimum there, and the gradient pushes toward it from both sides). At
training length that rate looks harmless (you keep ~95% of the signal). At 16× the
length, you've kept 45%, and the memory is useless — skill −6 versus a baseline
that ignores the input entirely, on 10/10 seeds, while the constrained version sits
at +1.000 (paired sign tests, p = 0.002, and yes I committed the tests to git
before running them, because of what happened in finding 0).

**2. The fix matters less than I wanted it to.** Three different things repair the
baseline: freezing the decay parameter at zero (which works for a mildly cursed
reason — the gradients get scaled below Adam's ε floor; I measured this too),
the hard projection, and — this one stung — just cranking regularization way up
(λ=30 rescues 3/3; my earlier "soft fixes don't work" claim came from only testing
λ=3, which fails, as does 3× more training, 2/5). So the honest statement is: the
*default configuration* fails, and anything that pins the effective decay at zero
fixes it.

**3. Where the fancy method actually earns its keep: when forgetting is
necessary.** On my first benchmark, "never forget anything" tied with my method,
which was embarrassing until I realized the benchmark never punished hoarding. So I
built one where it provably does (a memory that can't forget pays 13× on
current-content accuracy — I validated this *before* running the comparison), and
there the projection is the only arm that reliably holds the ledger: +1.000 on
10/10 seeds, beats every alternative on the pre-registered paired tests (p =
0.002–0.021), while the frozen-decay tricks drop the ledger about 1 seed in 10 and
hover at 0.97–0.99 otherwise. Content accuracy: no significant difference. So the
edge is exactness and reliability, not raw performance. I'll take it, but it's
narrower than I hoped.

**4. A prediction of mine that died: I was sure the attractor would track the
training horizon** (train on longer sequences → learn proportionally less
forgetting). Nope. Across an 8× range of training lengths it barely moves
(pre-registered threshold said refuted, so: refuted). The part that *does*
generalize is the failure itself — under-retention at 16× showed up at every
training length I tried. Why the attractor sits where it sits is genuinely open;
my sweep can't separate "set by the init" from "set by the task."

**Finding 0, which colors everything above:** my first headline result — "my
method is 29× better!!" — was fake. Not fraud, just a subtle rigging: I'd given
the baseline a fixed leak my method was exempt from, with a timescale that
happened to match my test length exactly. An adversarial audit (which I ran on
myself, and which hurt) caught it with a one-variable control: change one
constant, advantage becomes 1.00×. The retraction is permanently in the repo,
§5 of the dashboard, because the audit protocol that caught it ended up being one
of the most useful things the project produced.

## The experimental process, exactly

**The memory.** State `z ∈ R^8`, update `z_{k+1} = exp(W)·z_k + Enc(x_k)`. Enc and
Dec are small MLPs (two hidden layers, width 64). Ledger readout is
`gain·⟨r, z⟩ + bias` with `r` learned (tied to the conserved direction for the
pinned arm). The *only* difference between arms is the shape of `W`:

| arm | W | can it protect one direction? |
|---|---|---|
| scalar | `(A−Aᵀ) − softplus(s)·I` | no — one decay for everything |
| diag | `(A−Aᵀ) − diag(softplus(d))` | in principle; has to learn it |
| ks | `(A−Aᵀ) − BBᵀ − softplus(s)·I` | in principle; has to learn it |
| scalar0 / diag0 | same, but decay initialized at −20 (≈0, effectively frozen under Adam) | protects *everything* |
| pinned | `P(A−Aᵀ)P − (PB)(PB)ᵀ − softplus(s)·P`, `P = I − CCᵀ` | yes, exactly, by construction |

I counted live parameters with an autograd probe because an auditor called me on
it: scalar 75 / diag 82 / ks 139 / pinned 139 outside the encoder/decoder
(~13.6k, identical everywhere). ks matches pinned exactly, and the frozen
variants match their parents, so capacity doesn't explain any of the gaps.

**The task.** Streams of K blocks. Each block might contain a "deposit"
(flag × amount — both are channels *inside* the block, so the running-total label
is a nonlinear function of the input and no architecture gets it for free). The
deposit rate is drawn per-stream from U(0.1, 0.5) so you can't fake it with a
constant. Two variants: the original (32-dim blocks, 8-dim memory) and the
forced-forgetting one (8-dim blocks, high-variance content you must reconstruct,
target = the six non-deposit channels). For the second one I swept *fixed* decay
rates first to prove the tension exists — never-forgetting costs 13× on content
(criterion was ≥20% improvement from forgetting; it came in at 92.4%).

**Training.** Identical for every arm: Adam, lr 3e-3, batch 64, 3000 steps (2000
on the forgetting task), LR halved every third, grad clip 1.0, K_train = 64.
Loss = MSE on the ledger at every prefix + MSE on content reconstruction.
Weights seeded with `torch.manual_seed`, data from a separate seeded generator.

**Evaluation.** K_test = 1024, fresh data generator (disjoint seed; every arm at a
given seed sees the *identical* eval batch, which is what makes paired tests
legitimate). Skill = 1 − MSE/MSE_constant where the constant is the eval batch's
own mean — so 0 means "no better than ignoring the input" and 1 is perfect.
Content = reconstruction MSE at t=64 and t=1024.

**Statistics.** 10 seeds, paired, exact two-sided sign tests, comparisons
committed to git before the data existed (one earlier run was only partially
pre-registered that way; the docs say so and give the fresh-seed-only number).

**Mechanism probes.** Four: a 22-point sweep of the decay in a trained model
(finds the interior train-loss minimum right where training converged); per-decay
gradient statistics across 64 minibatches (the mean gradient pushes decay *up*
below the optimum, *down* above it — signal-to-noise 1.3–11.2, and the 1.3 sits
exactly at the optimum's zero crossing, which is what you'd expect); a per-step
Adam probe of the frozen-zero arm (updates collapse because √v̂ ≈ 2.7e-9 sits
under Adam's ε = 1e-8); and the training-length sweep from finding 4.

**Process stuff I now do religiously.** One-variable controls. A no-input floor
under every metric. Oracle recalibration on held-out data to separate "lost the
information" from "mis-scaled it." A bit-exact reproduction guard before extending
any experiment. Environment stamps (torch/CUDA/GPU/commit) in every log. And the
verifier script that re-reads all 64 quantitative claims from the raw logs —
which has caught me three separate times.

## What's wrong with it

- **It's all synthetic so far.** Tiny toy tasks built to isolate one effect. The
  real-LM version (same experiment over a frozen Qwen model's hidden states on
  real text) is literally running while I write this; it could absolutely come
  back "no transfer."
- **The guarantee only covers what you name in advance.** The ledger protects
  quantities you designate and route deposits into. "Remember whatever turns out
  to matter" is a different, harder problem this doesn't touch.
- **Cheap tricks get you most of the way.** A frozen parameter is one line and
  captures ~97–99% of the benefit unless forgetting pressure is real. My method
  wins on exactness and reliability, which you may or may not care about.
- **The mechanism story is narrower than it sounds.** Measured on one task
  family, mostly through 1-D probes of single models. What sets the attractor's
  location is unresolved.
- **Marginal effect sizes are small.** +1.000 vs ~0.97–0.99, 10/10 vs 9/10.
  Statistically real; practically, depends entirely on your stakes.

## Where it stands

Repo: github.com/mjkaiser2000/latent-memory — the dashboard (`dashboard.html`) has
the full research log including the retraction, interactive plots of the actual
data, and provenance links from every claim to the log file that backs it. Next
milestone: the real-LM run currently in progress, then (if that survives) swapping
this into an actual eviction-style LLM memory and testing at deployment lengths.
