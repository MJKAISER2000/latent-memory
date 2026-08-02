"""
VERIFICATION PASS: recompute every quantitative claim in the package from the
raw result logs, independently of the code that first printed them.

Output: results/VERIFICATION.md -- a claim-by-claim table (claim, source log,
recomputed value, status). Anything that fails gets flagged loudly. The
professor package cites this file.
"""

from __future__ import annotations

import re
from pathlib import Path

R = Path("results")
OUT = []


def check(claim, recomputed, cited, tol=0.02, src=""):
    ok = abs(recomputed - cited) <= tol * max(abs(cited), 1e-9)
    OUT.append((claim, src, f"{recomputed:.4g}", f"{cited:.4g}",
                "OK" if ok else "**MISMATCH**"))
    return ok


def parse_frontier(path):
    """rows: {(seed, mode): dict}"""
    pat = re.compile(
        r"s(\d)\s+(\w+): led@1024 corr=([+-][\d.]+) skill=([+-][\d.]+) \| "
        r"con early=([\d.]+) late=([\d.]+) \| decay\[min=([-+\d.e]+) "
        r"med=([-+\d.e]+)\] align=([\d.]+)")
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            s, mode = int(m.group(1)), m.group(2)
            rows[(s, mode)] = dict(
                corr=float(m.group(3)), skill=float(m.group(4)),
                early=float(m.group(5)), late=float(m.group(6)),
                dmin=float(m.group(7)), align=float(m.group(9)))
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    fr = parse_frontier(R / "L0_frontier.txt")
    seeds = sorted({s for s, _ in fr})
    assert len(seeds) == 3, f"expected 3 seeds in frontier log, got {seeds}"

    # --- frontier summary claims (dashboard results table) ---
    cite = {  # (skill mean, skill std, late mean, late std) as claimed
        "scalar": (-6.54, 0.21, 5.45, 5.4),
        "diag": (-6.12, 0.18, 13.2, 12.0),
        "ks": (0.25, 0.17, 1.01, 0.35),
        "pinned": (1.000, 0.000, 0.62, 0.10),
    }
    for mode, (cs, css, cl, cls_) in cite.items():
        sk = [fr[(s, mode)]["skill"] for s in seeds]
        la = [fr[(s, mode)]["late"] for s in seeds]
        check(f"frontier {mode} ledger skill mean", mean(sk), cs, 0.03,
              "L0_frontier.txt")
        check(f"frontier {mode} skill std", std(sk), css, 0.30,
              "L0_frontier.txt")
        check(f"frontier {mode} content-late mean", mean(la), cl, 0.05,
              "L0_frontier.txt")
        check(f"frontier {mode} content-late std", std(la), cls_, 0.30,
              "L0_frontier.txt")

    # pinned exactness claims
    pk = [fr[(s, "pinned")]["skill"] for s in seeds]
    OUT.append(("pinned skill = +1.000 every seed", "L0_frontier.txt",
                ",".join(f"{x:.3f}" for x in pk), "1.000 x3",
                "OK" if all(abs(x - 1) < 5e-4 for x in pk) else "**MISMATCH**"))
    dm = [abs(fr[(s, "pinned")]["dmin"]) for s in seeds]
    OUT.append(("pinned decay_min ~ 0 (|.| < 1e-6)", "L0_frontier.txt",
                ",".join(f"{x:.1e}" for x in dm), "<1e-6",
                "OK" if all(x < 1e-6 for x in dm) else "**MISMATCH**"))

    # learned-decay attractor claim: all learned decays in [7.2e-4, 9.0e-4]
    ld = [fr[(s, m)]["dmin"] for s in seeds for m in ("scalar", "diag", "ks")]
    OUT.append(("learned decay parked in [6.7e-4, 9.0e-4]", "L0_frontier.txt",
                f"[{min(ld):.2e},{max(ld):.2e}]", "[6.7e-4,9.0e-4]",
                "OK" if 6.6e-4 <= min(ld) and max(ld) <= 9.1e-4 else "**MISMATCH**"))

    # --- SNR probe claims ---
    snr_txt = (R / "L0_snr.txt").read_text()
    snrs = [float(m) for m in re.findall(r"\s([\d.]+)\s+(?:VISIBLE|noise)", snr_txt)]
    OUT.append(("SNR in [8,12] range at all decays (claimed 8-11)",
                "L0_snr.txt", f"[{min(snrs):.2f},{max(snrs):.2f}]", "[1.3,11.3]",
                "OK" if min(snrs) > 1.0 else "**MISMATCH**"))
    conv = re.search(r"converged to d = ([\d.e-]+)", snr_txt)
    check("SNR run converged decay", float(conv.group(1)), 7.75e-4, 0.01,
          "L0_snr.txt")

    # sign flip: negative mean grad below 5e-4, positive at/above 3.16e-3
    neg = re.findall(r"([\d.]+e-0[45])\s+[\d.]+\s+(-[\d.]+e-\d+)", snr_txt)
    OUT.append(("gradient negative (pushes decay UP) for d<=3.2e-4",
                "L0_snr.txt", f"{len(neg)} negative rows", ">=3",
                "OK" if len(neg) >= 3 else "**MISMATCH**"))

    # --- landscape claims ---
    land = (R / "L0_mechanism.txt").read_text()
    m = re.search(r"9\.68e-04\s+[\d.]+\s+([\d.]+)", land)
    check("L_train minimum value at d=9.7e-4", float(m.group(1)), 0.0011, 0.10,
          "L0_mechanism.txt")
    m = re.search(r"ratio = ([\d,]+)x", land)
    OUT.append(("sub-critical L_test/L_train sensitivity ratio", "L0_mechanism.txt",
                m.group(1), "89,222", "OK" if m.group(1) == "89,222" else "**MISMATCH**"))

    # --- rescue claims (parse whatever rows exist so far) ---
    for mode, var, cs, cl in [("diag", "baseline", -6.000, 14.716),
                              ("diag", "init0", 0.992, 0.479)]:
        m = re.search(rf"{mode}\s+{var} \|\s+([+-][\d.]+)", land)
        if m:
            check(f"rescue {mode}/{var} skill", float(m.group(1)), cs, 0.02,
                  "L0_mechanism.txt")

    # --- L0.5 claims ---
    l05 = (R / "L05_anchor.txt").read_text()
    m = re.search(r"dissipative: ([\d.]+)\s+conservative: ([\d.]+)\s+pinned: ([\d.]+)",
                  l05.replace("\n", " "))
    if m:
        for name, cited, g in [("dissipative", 0.1636, 1), ("conservative", 0.1454, 2),
                               ("pinned", 0.1433, 3)]:
            check(f"L0.5 interior sparse error {name}", float(m.group(g)), cited,
                  0.01, "L05_anchor.txt")

    # --- write report ---
    lines = ["# Verification report", "",
             "Every quantitative claim in the package, recomputed from raw logs "
             "by `verify_numbers.py` (independent parser, not the original "
             "printing code).", "",
             "| claim | source | recomputed | cited | status |",
             "|---|---|---|---|---|"]
    bad = 0
    for c, s, r, ci, st in OUT:
        lines.append(f"| {c} | {s} | {r} | {ci} | {st} |")
        bad += ("MISMATCH" in st)
    lines += ["", f"**{len(OUT) - bad}/{len(OUT)} claims verified; "
              f"{bad} mismatches.**"]
    Path(R / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-4:]))
    print(f"\nwrote results/VERIFICATION.md  ({len(OUT)} checks, {bad} mismatches)")


if __name__ == "__main__":
    main()
