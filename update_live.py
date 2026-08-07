"""
Regenerates `live.js` -- the dashboard's dynamic data feed.

Why a .js file and not fetch()/JSON: the dashboard is opened via file://, where
fetch() of local files is blocked by CORS but <script src> injection is not.
The dashboard re-injects live.js?bust=<ts> every 30 s and re-renders its LIVE
strip, so a browser tab left open tracks running experiments.

Usage:
    python update_live.py            # write once
    python update_live.py --watch    # rewrite every 20 s (run during long jobs)

Everything is parsed from the results/ logs -- same provenance rule as the
rest of the dashboard: if it is not in a log, it is not shown.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
R = ROOT / "results"

STATS_EXPECTED = 60          # 6 arms x 10 seeds


def parse_stats():
    """Progress + rolling rows + final summary of the n=10 run."""
    p = R / "L0_stats.txt"
    if not p.exists():
        return {"status": "not started", "done": 0, "expected": STATS_EXPECTED,
                "last_rows": [], "summary": None, "tests": []}
    txt = p.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"s(\d)\s+(\w+): skill=([+-][\d.]+) late=([\d.]+) "
                      r"dmin=([\d.e+-]+)", txt)
    done = len(rows)
    last = [f"s{s} {m}: skill {sk}, content {la}" for s, m, sk, la, _ in rows[-4:]]
    summary = None
    if "MEDIAN [IQR]" in txt:
        med = re.findall(r"^\s*(\w+) \|\s+([+-][\d.]+) \[([^\]]+)\] \|\s+([\d.]+)",
                         txt, re.M)
        summary = [{"arm": a, "skill_med": sm, "skill_iqr": iq, "late_med": lm}
                   for a, sm, iq, lm in med]
    tests = re.findall(r"(\w+) vs\s+(\w+) \(([^)]+)\): wins (\d+)/(\d+), "
                       r"sign-test p=([\d.e-]+)", txt)
    status = ("complete" if summary is not None
              else f"running — {done}/{STATS_EXPECTED} configs")
    return {"status": status, "done": done, "expected": STATS_EXPECTED,
            "last_rows": last, "summary": summary,
            "tests": [{"a": a, "b": b, "what": w, "wins": f"{x}/{n}", "p": p_}
                      for a, b, w, x, n, p_ in tests]}


def run_table():
    """Status of every experiment log the dashboard cites."""
    runs = []

    def add(name, path, complete_marker, note=""):
        p = (R / path) if not path.startswith("..") else (ROOT / path[3:])
        if not p.exists():
            runs.append({"name": name, "state": "pending", "note": note})
            return
        txt = p.read_text(encoding="utf-8", errors="replace")
        state = "done" if complete_marker in txt else "running"
        runs.append({"name": name, "state": state, "note": note})

    add("L0 frontier (3 seeds)", "L0_frontier.txt", "s2  pinned")
    add("SNR probe", "L0_snr.txt", "IMPLIED DRIFT")
    add("Landscape + rescue suite", "L0_mechanism.txt", "pinned  baseline")
    add("L0.5 anchor transport", "L05_anchor.txt", "interior (t<64")
    add("Retraction control (regenerated)", "adv_control.txt", "pinned(e=0.001)   1024")
    add("Freeze probe", "freeze_probe.txt", "VERDICT")
    add("L0-Forget (necessity + arms)", "L0_forget.txt", "READING")
    add("Verification pass", "VERIFICATION.md", "claims verified")
    st = parse_stats()
    runs.append({"name": "n=10 statistics (6 arms)",
                 "state": "done" if st["summary"] else
                          ("running" if st["done"] else "pending"),
                 "note": st["status"]})
    return runs, st


def verification_count():
    """Parse the live count from VERIFICATION.md (hardening item 8: the count
    staled three times when maintained by hand)."""
    p = R / "VERIFICATION.md"
    if not p.exists():
        return None
    m = re.search(r"\*\*(\d+)/(\d+) claims verified; (\d+) mismatch",
                  p.read_text(encoding="utf-8", errors="replace"))
    return {"ok": m.group(1), "total": m.group(2),
            "mismatches": m.group(3)} if m else None


def build():
    runs, stats = run_table()
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
        "stats": stats,
        "verif": verification_count(),
    }
    (ROOT / "live.js").write_text(
        "window.LIVE = " + json.dumps(payload, indent=1) + ";\n"
        "if (window.renderLive) window.renderLive();\n",
        encoding="utf-8")
    return payload


if __name__ == "__main__":
    if "--watch" in sys.argv:
        print("watching; ctrl-c to stop")
        while True:
            p = build()
            print(f"  {p['generated']}  stats: {p['stats']['status']}", flush=True)
            time.sleep(20)
    else:
        p = build()
        print(f"wrote live.js  ({p['generated']})  stats: {p['stats']['status']}")
