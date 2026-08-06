window.LIVE = {
 "generated": "2026-08-06 02:27:00",
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
   "name": "Retraction control (regenerated)",
   "state": "done",
   "note": ""
  },
  {
   "name": "Freeze probe",
   "state": "done",
   "note": ""
  },
  {
   "name": "L0-Forget (necessity + arms)",
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
   "state": "done",
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
  "summary": [
   {
    "arm": "scalar",
    "skill_med": "-6.293",
    "skill_iqr": "-6.538,-6.054",
    "late_med": "5.522"
   },
   {
    "arm": "diag",
    "skill_med": "-5.702",
    "skill_iqr": "-6.246,-5.263",
    "late_med": "5.452"
   },
   {
    "arm": "ks",
    "skill_med": "+0.223",
    "skill_iqr": "-0.085,+0.456",
    "late_med": "1.108"
   },
   {
    "arm": "diag0",
    "skill_med": "+0.998",
    "skill_iqr": "+0.988,+1.000",
    "late_med": "0.540"
   },
   {
    "arm": "scalar0",
    "skill_med": "+1.000",
    "skill_iqr": "+0.991,+1.000",
    "late_med": "0.332"
   },
   {
    "arm": "pinned",
    "skill_med": "+1.000",
    "skill_iqr": "+1.000,+1.000",
    "late_med": "0.758"
   }
  ],
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
