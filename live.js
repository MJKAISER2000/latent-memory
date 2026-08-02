window.LIVE = {
 "generated": "2026-08-02 17:38:13",
 "runs": [
  {
   "name": "L0 frontier (3 seeds)",
   "state": "done",
   "note": ""
  },
  {
   "name": "SNR probe",
   "state": "done",
   "note": ""
  },
  {
   "name": "Landscape + rescue suite",
   "state": "done",
   "note": ""
  },
  {
   "name": "L0.5 anchor transport",
   "state": "done",
   "note": ""
  },
  {
   "name": "Verification pass",
   "state": "done",
   "note": ""
  },
  {
   "name": "n=10 statistics (6 arms)",
   "state": "running",
   "note": "complete"
  }
 ],
 "stats": {
  "status": "complete",
  "done": 60,
  "expected": 60,
  "last_rows": [
   "s9 ks: skill +0.369, content 0.639",
   "s9 diag0: skill +1.000, content 0.283",
   "s9 scalar0: skill +1.000, content 0.300",
   "s9 pinned: skill +0.999, content 0.840"
  ],
  "summary": [],
  "tests": [
   {
    "a": "pinned",
    "b": "scalar",
    "what": "superiority",
    "wins": "10/10",
    "p": "0.001953"
   },
   {
    "a": "pinned",
    "b": "diag",
    "what": "superiority",
    "wins": "10/10",
    "p": "0.001953"
   },
   {
    "a": "pinned",
    "b": "ks",
    "what": "superiority",
    "wins": "10/10",
    "p": "0.001953"
   },
   {
    "a": "pinned",
    "b": "diag0",
    "what": " fix-parity",
    "wins": "7/10",
    "p": "0.3438"
   },
   {
    "a": "pinned",
    "b": "scalar0",
    "what": " fix-parity",
    "wins": "7/10",
    "p": "0.3438"
   },
   {
    "a": "scalar0",
    "b": "scalar",
    "what": "  init0-fix",
    "wins": "10/10",
    "p": "0.001953"
   }
  ]
 }
};
if (window.renderLive) window.renderLive();
