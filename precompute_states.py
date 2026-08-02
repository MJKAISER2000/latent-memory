"""
Stage 1 of the L1 pipeline: run the frozen LM over haystack samples once and
dump block-pooled hidden states to disk.

Why precompute: the memory module is tiny (<1M params) but training it needs
many epochs over long sequences. Running the LM every epoch is wasteful and,
on a 6 GB card, the binding constraint. Dump once, then train the memory module
on cached tensors -- at which point VRAM stops mattering and the same cache
transfers unchanged to the cluster.

CONTEXT MODE -- a real design decision, not a performance knob:

  local  (default)  Each chunk is encoded independently. Block representations
                    see only their own chunk. This is the HONEST setting for
                    our hypothesis: we are asking whether the *memory module*
                    can carry long-range information, so the encoder must not
                    smuggle it in through full attention.

  full              Chunks attend to all previous chunks via KV cache. Block
                    representations are contextualised by the whole prefix.
                    Costs O(n^2) and lets the LM itself do the retention, which
                    confounds the measurement. Use only as a topline reference.

If you report L1 numbers, say which mode. They are not comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_lm(model_id: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def encode_blocks(
    ids: list[int],
    model,
    device: str,
    layer: int,
    block: int,
    chunk: int,
    context_mode: str,
):
    """Mean-pool hidden states within each block of `block` tokens.

    Returns (n_blocks, d_model) float32 on CPU.
    """
    out = []
    past = None
    for start in range(0, len(ids), chunk):
        piece = torch.tensor(ids[start:start + chunk], device=device)[None]
        kw = {"output_hidden_states": True, "use_cache": context_mode == "full"}
        if context_mode == "full" and past is not None:
            kw["past_key_values"] = past
        res = model(piece, **kw)
        if context_mode == "full":
            past = res.past_key_values
        h = res.hidden_states[layer][0].float()          # (chunk_len, d_model)

        # Pool into blocks. Blocks are aligned to absolute token position so
        # that a block never straddles a chunk boundary inconsistently.
        for b0 in range(0, h.shape[0], block):
            out.append(h[b0:b0 + block].mean(dim=0))
        del res
    return torch.stack(out).cpu()


def main():
    ap = argparse.ArgumentParser(description="Cache block-pooled LM states.")
    ap.add_argument("--jsonl", required=True, help="haystack .jsonl from data/haystack.py")
    ap.add_argument("--out", required=True, help="output .pt shard")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer", type=int, default=-2,
                    help="hidden_states index; -2 = penultimate")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="tokens per forward pass; lower this if you OOM")
    ap.add_argument("--context-mode", default="local", choices=["local", "full"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32", action="store_true", help="use fp32 (slower, more VRAM)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.fp32 else torch.float16
    if device == "cpu":
        dtype = torch.float32

    tok, model = load_lm(args.model, device, dtype)
    print(f"loaded {args.model} on {device} ({dtype}), "
          f"layer={args.layer} block={args.block} mode={args.context_mode}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data.haystack import Sample, tokenize_sample

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    recs = []
    for i, r in enumerate(rows):
        s = Sample(**r)
        t = tokenize_sample(s, tok, block=args.block)
        X = encode_blocks(t["input_ids"], model, device, args.layer,
                          args.block, args.chunk, args.context_mode)
        mask = torch.tensor(t["durable_block_mask"][: X.shape[0]], dtype=torch.bool)
        if mask.shape[0] < X.shape[0]:                 # pad if rounding differed
            mask = torch.cat([mask, torch.zeros(X.shape[0] - mask.shape[0],
                                                dtype=torch.bool)])
        recs.append({
            "X": X,                                    # (n_blocks, d_model)
            "durable": mask,                           # (n_blocks,)
            "query_ids": torch.tensor(t["query_ids"]),
            "answer": t["answer"],
            "answer_ids": torch.tensor(t["answer_ids"]),
            "task": s.task,
            "n_tokens": t["n_tokens"],
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(rows)}] {t['n_tokens']:,} tok -> "
                  f"{X.shape[0]} blocks x {X.shape[1]}", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": recs,
        "d_model": recs[0]["X"].shape[1],
        "config": vars(args),
    }, outp)

    mb = outp.stat().st_size / 1e6
    nb = sum(r["X"].shape[0] for r in recs) / len(recs)
    print(f"\nwrote {len(recs)} records -> {outp}  ({mb:.1f} MB)")
    print(f"  d_model={recs[0]['X'].shape[1]}  mean blocks/sample={nb:.0f}")
    print(f"  durable blocks/sample={sum(r['durable'].sum().item() for r in recs)/len(recs):.1f}")


if __name__ == "__main__":
    main()
