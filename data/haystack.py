"""
Synthetic needle-in-haystack corpus for the latent twin memory experiments.

Designed around the specific hypothesis under test, not around generic
long-context benchmarking. The pinned-generator claim is:

    designated content survives in the conserved subspace while everything
    orthogonal to it decays monotonically

so the corpus must supply a *labelled* split between durable content and
disposable content, and must measure retention as a function of distance.

Three tasks, in increasing order of how much they favour the balance-law view:

  retrieve  A fact is stated once, then N tokens of filler, then queried.
            The classic needle test. Measures raw retention vs. distance.

  update    A fact is stated, then REVISED one or more times, then queried.
            The answer is the latest value. This is the balance-law task:
            each revision is a flux into the ledger, and the correct readout
            is the accumulated state, not any single mention.

  count     A quantity is deposited repeatedly; the query asks for the total.
            Directly the L0 ledger task in natural language. Nothing in the
            context states the answer -- it must be accumulated.

Distractors matter. Every needle has surface-form twins scattered through the
filler (same sentence template, different entity), so a model cannot win by
pattern-matching the template. Without this the task is trivially solvable by
attending to "the access code for X is" and the retention curve is meaningless.

Tokenizer-agnostic: emits text plus character offsets. Call `tokenize_sample`
once a tokenizer is available to get block-aligned token ids and per-block
labels.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Surface inventory
# ---------------------------------------------------------------------------

ENTITIES = [
    "Vermillion", "Kestrel", "Ostrander", "Halcyon", "Brightwater", "Calloway",
    "Dunmore", "Ellsworth", "Fairbanks", "Glenridge", "Harrowgate", "Ingleside",
    "Jessup", "Kirkwall", "Lindholm", "Marchetti", "Northgate", "Oakhurst",
    "Pemberton", "Quillon", "Ravenscar", "Stonebridge", "Thornbury", "Uxbridge",
    "Valebrook", "Westmarch", "Yarrowfield", "Ziegler", "Ashgrove", "Blackmoor",
]

ATTRIBUTES = [
    ("access code", "vault"), ("routing number", "branch"),
    ("serial number", "turbine"), ("badge id", "lab"),
    ("locker code", "depot"), ("clearance level", "wing"),
    ("manifest number", "shipment"), ("checksum", "archive"),
]

FILLER_TEMPLATES = [
    "The {e} {n} was inspected on a routine schedule and no anomalies were noted.",
    "Maintenance staff rotated through the {e} {n} without incident this quarter.",
    "A quarterly summary for the {e} {n} was filed with the regional office.",
    "The {e} {n} remains listed under the standard operating classification.",
    "Personnel assigned to the {e} {n} completed the annual refresher module.",
    "Environmental readings near the {e} {n} stayed within the expected band.",
    "The logistics team confirmed that the {e} {n} is operating nominally.",
    "No changes to the {e} {n} were requested during the review window.",
    "Documentation for the {e} {n} was migrated to the updated record system.",
    "The {e} {n} appeared in the consolidated index without exception flags.",
]

NEEDLE_TEMPLATE = "The {attr} for the {e} {noun} is {val}."
UPDATE_TEMPLATE = "Correction: the {attr} for the {e} {noun} is now {val}."
DEPOSIT_TEMPLATE = "A shipment of {val} units was logged at the {e} {noun}."

QUERY = {
    "retrieve": "What is the {attr} for the {e} {noun}?",
    "update": "What is the current {attr} for the {e} {noun}?",
    "count": "How many units in total were logged at the {e} {noun}?",
}


# ---------------------------------------------------------------------------
# Config / sample
# ---------------------------------------------------------------------------


@dataclass
class HaystackConfig:
    task: str = "retrieve"              # retrieve | update | count
    n_filler: int = 2000                # filler sentences (~15 tok each)
    n_needles: int = 1                  # distinct facts to plant
    n_updates: int = 3                  # revisions per fact (task="update")
    n_deposits: int = 8                 # deposits per entity (task="count")
    depths: tuple = (0.1,)              # fractional positions of the needles
    depth_random: bool = False          # sample depth per-sample instead
    depth_range: tuple = (0.02, 0.90)   # range used when depth_random
    distractor_ratio: float = 0.25      # fraction of filler that mimics needles
    seed: int = 0


@dataclass
class Sample:
    text: str
    query: str
    answer: str
    task: str
    # char offsets of content that MUST survive -> these become pinned blocks
    durable_spans: list = field(default_factory=list)
    # distance from last durable mention to the query, in characters
    needle_char_distance: int = 0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _filler_sentence(rng, used_entity: str, attr_pool, as_distractor: bool,
                     task: str = "retrieve"):
    """A filler line.

    Distractors must mimic the template used by THIS TASK's durable statements,
    with a wrong entity. Otherwise the durable sentences are the only ones of
    their surface form and the task collapses to template matching -- the
    retention curve then measures nothing.
    """
    e = rng.choice([x for x in ENTITIES if x != used_entity])
    if as_distractor:
        attr, noun = rng.choice(attr_pool)
        val = str(rng.randint(10000, 99999))
        if task == "count":
            return DEPOSIT_TEMPLATE.format(val=rng.randint(1, 40), e=e, noun=noun)
        if task == "update":
            # mix plain statements and corrections so "find the last
            # 'Correction:'" is not a winning strategy
            tmpl = UPDATE_TEMPLATE if rng.random() < 0.5 else NEEDLE_TEMPLATE
            return tmpl.format(attr=attr, e=e, noun=noun, val=val)
        return NEEDLE_TEMPLATE.format(attr=attr, e=e, noun=noun, val=val)
    t = rng.choice(FILLER_TEMPLATES)
    _, noun = rng.choice(attr_pool)
    return t.format(e=e, n=noun)


def make_sample(cfg: HaystackConfig, rng: random.Random) -> Sample:
    entity = rng.choice(ENTITIES)
    attr, noun = rng.choice(ATTRIBUTES)

    # Build the durable statements for this task.
    durable: list[str] = []
    if cfg.task == "retrieve":
        val = str(rng.randint(10000, 99999))
        durable = [NEEDLE_TEMPLATE.format(attr=attr, e=entity, noun=noun, val=val)]
        answer = val

    elif cfg.task == "update":
        vals = [str(rng.randint(10000, 99999)) for _ in range(cfg.n_updates + 1)]
        durable = [NEEDLE_TEMPLATE.format(attr=attr, e=entity, noun=noun, val=vals[0])]
        durable += [UPDATE_TEMPLATE.format(attr=attr, e=entity, noun=noun, val=v)
                    for v in vals[1:]]
        answer = vals[-1]          # latest value wins

    elif cfg.task == "count":
        amounts = [rng.randint(1, 40) for _ in range(cfg.n_deposits)]
        durable = [DEPOSIT_TEMPLATE.format(val=a, e=entity, noun=noun) for a in amounts]
        answer = str(sum(amounts))  # stated NOWHERE in the context
    else:
        raise ValueError(f"unknown task {cfg.task!r}")

    # Filler stream.
    filler = [
        _filler_sentence(rng, entity, ATTRIBUTES,
                         as_distractor=rng.random() < cfg.distractor_ratio,
                         task=cfg.task)
        for _ in range(cfg.n_filler)
    ]

    # Place durable statements at the requested fractional depths.
    if cfg.depth_random:
        # One depth drawn per sample. Binning by the realised needle->query
        # distance at eval time then yields a continuous retention curve from a
        # single dataset, instead of one dataset per depth.
        lo, hi = cfg.depth_range
        d0 = lo + (hi - lo) * rng.random()
        if len(durable) == 1:
            depths = [d0]
        else:
            span = min(hi - d0, 0.8)
            depths = [d0 + span * i / (len(durable) - 1) for i in range(len(durable))]
    elif len(cfg.depths) >= len(durable):
        depths = list(cfg.depths)[:len(durable)]
    else:
        # spread remaining statements evenly after the first given depth
        d0 = cfg.depths[0]
        depths = [d0 + (0.8 - d0) * i / max(len(durable) - 1, 1)
                  for i in range(len(durable))]

    slots = sorted({min(int(d * len(filler)), len(filler)) for d in depths})
    while len(slots) < len(durable):                 # de-duplicated collisions
        slots = sorted(set(slots) | {min(slots[-1] + 1, len(filler))})
    slots = slots[:len(durable)]

    lines, spans, di = [], [], 0
    for i in range(len(filler) + 1):
        while di < len(durable) and di < len(slots) and slots[di] == i:
            start = sum(len(s) + 1 for s in lines)
            lines.append(durable[di])
            spans.append((start, start + len(durable[di])))
            di += 1
        if i < len(filler):
            lines.append(filler[i])

    text = "\n".join(lines)
    query = QUERY[cfg.task].format(attr=attr, e=entity, noun=noun)
    dist = len(text) - spans[-1][1] if spans else 0

    return Sample(
        text=text, query=query, answer=answer, task=cfg.task,
        durable_spans=spans, needle_char_distance=dist,
        meta={"entity": entity, "attr": attr, "noun": noun,
              "n_durable": len(durable), "n_filler": cfg.n_filler},
    )


def make_dataset(cfg: HaystackConfig, n: int) -> list[Sample]:
    rng = random.Random(cfg.seed)
    return [make_sample(cfg, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tokenisation -> block-aligned tensors with per-block durability labels
# ---------------------------------------------------------------------------


def tokenize_sample(sample: Sample, tokenizer, block: int = 64):
    """Returns dict with input_ids, block-level durable mask, and answer ids.

    `durable_block_mask[k] == True` iff block k overlaps a durable span. That
    mask is the supervision signal for which blocks should route into the
    pinned subspace.
    """
    enc = tokenizer(sample.text, return_offsets_mapping=True,
                    add_special_tokens=False)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]

    n_blocks = (len(ids) + block - 1) // block
    mask = [False] * n_blocks
    for (s, e) in sample.durable_spans:
        for ti, (ts, te) in enumerate(offs):
            if ts < e and te > s:
                mask[ti // block] = True

    return {
        "input_ids": ids,
        "n_blocks": n_blocks,
        "durable_block_mask": mask,
        "query_ids": tokenizer(sample.query, add_special_tokens=False)["input_ids"],
        "answer_ids": tokenizer(sample.answer, add_special_tokens=False)["input_ids"],
        "answer": sample.answer,
        "n_tokens": len(ids),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Synthesize haystack corpora.")
    ap.add_argument("--out", default="data/haystack")
    ap.add_argument("--task", default="retrieve", choices=["retrieve", "update", "count"])
    ap.add_argument("--n", type=int, default=200, help="samples")
    ap.add_argument("--n-filler", type=int, default=2000)
    ap.add_argument("--depths", default="0.1", help="comma-separated fractions")
    ap.add_argument("--depth-random", action="store_true",
                    help="sample needle depth per-sample -> continuous distance curve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None,
                    help="HF id, e.g. Qwen/Qwen2.5-0.5B. Omit for text-only.")
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args()

    cfg = HaystackConfig(
        task=args.task, n_filler=args.n_filler, seed=args.seed,
        depths=tuple(float(x) for x in args.depths.split(",")),
        depth_random=args.depth_random,
    )
    ds = make_dataset(cfg, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for s in ds:
            f.write(json.dumps(asdict(s)) + "\n")

    chars = sum(len(s.text) for s in ds) / len(ds)
    print(f"wrote {len(ds)} samples -> {jsonl}")
    print(f"  task={cfg.task}  mean chars/sample={chars:,.0f} "
          f"(~{chars / 4:,.0f} tokens)")
    print(f"  mean needle->query distance = "
          f"{sum(s.needle_char_distance for s in ds) / len(ds):,.0f} chars")
    print(f"\n--- sample 0 (head/tail) ---")
    s = ds[0]
    print(s.text[:220].replace("\n", " | "))
    print("   ...")
    print(s.text[-220:].replace("\n", " | "))
    print(f"\nQ: {s.query}\nA: {s.answer}")
    print(f"durable spans: {s.durable_spans}")

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        t = tokenize_sample(ds[0], tok, block=args.block)
        print(f"\ntokenized with {args.tokenizer}: {t['n_tokens']:,} tokens, "
              f"{t['n_blocks']} blocks of {args.block}")
        print(f"  durable blocks: "
              f"{[i for i, m in enumerate(t['durable_block_mask']) if m]}")


if __name__ == "__main__":
    main()
