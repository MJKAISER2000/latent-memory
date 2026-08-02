# Verification report

Every quantitative claim in the package, recomputed from raw logs by `verify_numbers.py` (independent parser, not the original printing code).

| claim | source | recomputed | cited | status |
|---|---|---|---|---|
| frontier scalar ledger skill mean | L0_frontier.txt | -6.537 | -6.54 | OK |
| frontier scalar skill std | L0_frontier.txt | 0.2108 | 0.21 | OK |
| frontier scalar content-late mean | L0_frontier.txt | 5.452 | 5.45 | OK |
| frontier scalar content-late std | L0_frontier.txt | 5.324 | 5.4 | OK |
| frontier diag ledger skill mean | L0_frontier.txt | -6.124 | -6.12 | OK |
| frontier diag skill std | L0_frontier.txt | 0.1767 | 0.18 | OK |
| frontier diag content-late mean | L0_frontier.txt | 13.16 | 13.2 | OK |
| frontier diag content-late std | L0_frontier.txt | 11.93 | 12 | OK |
| frontier ks ledger skill mean | L0_frontier.txt | 0.251 | 0.25 | OK |
| frontier ks skill std | L0_frontier.txt | 0.1716 | 0.17 | OK |
| frontier ks content-late mean | L0_frontier.txt | 1.01 | 1.01 | OK |
| frontier ks content-late std | L0_frontier.txt | 0.3472 | 0.35 | OK |
| frontier pinned ledger skill mean | L0_frontier.txt | 1 | 1 | OK |
| frontier pinned skill std | L0_frontier.txt | 0 | 0 | OK |
| frontier pinned content-late mean | L0_frontier.txt | 0.6157 | 0.62 | OK |
| frontier pinned content-late std | L0_frontier.txt | 0.1015 | 0.1 | OK |
| pinned skill = +1.000 every seed | L0_frontier.txt | 1.000,1.000,1.000 | 1.000 x3 | OK |
| pinned decay_min ~ 0 (|.| < 1e-6) | L0_frontier.txt | 1.2e-07,8.1e-08,6.5e-08 | <1e-6 | OK |
| learned decay parked in [6.7e-4, 9.0e-4] | L0_frontier.txt | [6.73e-04,9.01e-04] | [6.7e-4,9.0e-4] | OK |
| SNR in [8,12] range at all decays (claimed 8-11) | L0_snr.txt | [1.31,11.23] | [1.3,11.3] | OK |
| SNR run converged decay | L0_snr.txt | 0.0007749 | 0.000775 | OK |
| gradient negative (pushes decay UP) for d<=3.2e-4 | L0_snr.txt | 4 negative rows | >=3 | OK |
| L_train minimum value at d=9.7e-4 | L0_mechanism.txt | 0.0011 | 0.0011 | OK |
| sub-critical L_test/L_train sensitivity ratio | L0_mechanism.txt | 89,222 | 89,222 | OK |
| rescue diag/baseline skill | L0_mechanism.txt | -6 | -6 | OK |
| rescue diag/init0 skill | L0_mechanism.txt | 0.992 | 0.992 | OK |
| L0.5 interior sparse error dissipative | L05_anchor.txt | 0.1636 | 0.1636 | OK |
| L0.5 interior sparse error conservative | L05_anchor.txt | 0.1454 | 0.1454 | OK |
| L0.5 interior sparse error pinned | L05_anchor.txt | 0.1433 | 0.1433 | OK |

**29/29 claims verified; 0 mismatches.**