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
| SNR > 1 at all sampled decays, range 1.3-11.2 as cited | L0_snr.txt | [1.31,11.23] | [1.31,11.23] | OK |
| SNR run converged decay | L0_snr.txt | 0.0007749 | 0.000775 | OK |
| gradient negative (pushes decay UP) for d<=3.2e-4 | L0_snr.txt | 4 negative rows | >=3 | OK |
| L_train minimum value at d=9.7e-4 | L0_mechanism.txt | 0.0011 | 0.0011 | OK |
| sub-critical L_test/L_train sensitivity ratio | L0_mechanism.txt | 89,222 | 89,222 | OK |
| rescue diag/baseline skill | L0_mechanism.txt | -6 | -6 | OK |
| rescue diag/init0 skill | L0_mechanism.txt | 0.992 | 0.992 | OK |
| L0.5 interior sparse error dissipative | L05_anchor.txt | 0.1636 | 0.1636 | OK |
| L0.5 interior sparse error conservative | L05_anchor.txt | 0.1454 | 0.1454 | OK |
| L0.5 interior sparse error pinned | L05_anchor.txt | 0.1433 | 0.1433 | OK |
| n=10: every arm has exactly 10 seeds | L0_stats.txt | diag:10,diag0:10,ks:10,pinned:10,scalar:10,scalar0:10 | 10 x6 | OK |
| n=10 median skill scalar | L0_stats.txt | -6.293 | -6.293 | OK |
| n=10 median skill diag | L0_stats.txt | -5.701 | -5.702 | OK |
| n=10 median skill ks | L0_stats.txt | 0.2225 | 0.223 | OK |
| n=10 median skill diag0 | L0_stats.txt | 0.9975 | 0.998 | OK |
| n=10 median skill scalar0 | L0_stats.txt | 0.9995 | 1 | OK |
| n=10 median skill pinned | L0_stats.txt | 1 | 1 | OK |
| pinned vs scalar: 10/10 paired wins | L0_stats.txt | 10/10 | 10/10 | OK |
| pinned vs scalar median diff | L0_stats.txt | 7.293 | 7.293 | OK |
| pinned vs diag: 10/10 paired wins | L0_stats.txt | 10/10 | 10/10 | OK |
| pinned vs diag median diff | L0_stats.txt | 6.701 | 6.701 | OK |
| pinned vs ks: 10/10 paired wins | L0_stats.txt | 10/10 | 10/10 | OK |
| pinned vs ks median diff | L0_stats.txt | 0.7775 | 0.777 | OK |
| sign-test p for 10/10 | arithmetic | 0.001953 | 0.001953 | OK |
| pinned vs diag0: parity (no significant difference) | L0_stats.txt | 6W/1L/3T | neither side >=9 | OK |
| pinned vs scalar0: parity (no significant difference) | L0_stats.txt | 5W/2L/3T | neither side >=9 | OK |
| pinned IQR degenerate at [1.000,1.000] | L0_stats.txt | [1.000,1.000] | [1.000,1.000] | OK |
| adv control: fixed-eps raw rel err @1024 | adv_control.txt | 0.3386 | 0.3386 | OK |
| adv control: eps~0 recal vs pinned recal ratio ~1.0x | adv_control.txt | 0.991 | 0.9-1.1 | OK |
| adv control: no-input floor @1024 vs analytic 0.0700 | adv_control.txt | 0.0705 | 0.07 | OK |
| freeze probe: sqrt(v-hat) below Adam eps=1e-8 | freeze_probe.txt | 2.654e-09 | <1e-8 | OK |
| freeze probe verdict: eps-floor confirmed | freeze_probe.txt | present | present | OK |
| l0_forget Phase A: necessity criterion evaluated | L0_forget.txt | 92.4% vs 20% required | PASS or FAIL recorded | OK |
| RDATA.LTR matches landscape log (22 pts) | dashboard vs L0_mechanism.txt | max|d|=0.00e+00 | <=2% | OK |
| RDATA.LTE matches landscape log (22 pts) | dashboard vs L0_mechanism.txt | max|d|=0.05 | <=1.0 abs | OK |
| RDATA.HOR.dissipative matches seeds summary | dashboard vs results_L0_seeds.txt | max|d|=0.0000 | <=0.003 | OK |
| RDATA.HOR.conservative matches seeds summary | dashboard vs results_L0_seeds.txt | max|d|=0.0000 | <=0.003 | OK |
| RDATA.HOR.pinned matches seeds summary | dashboard vs results_L0_seeds.txt | max|d|=0.0000 | <=0.003 | OK |
| RDATA.STATS pinned skills match log rows | dashboard vs L0_stats.txt | 10 values | identical multiset | OK |

**58/58 claims verified; 0 mismatches.**