# Pre-registered test plans

Each plan below was written into the build log (`NOTES.md`, untracked since 30 Oct 2026)
and committed before any of the test evaluations it covers ran. The text is copied
verbatim from the commit that introduced it; `git show <commit>:NOTES.md` shows the
original in place.

## Runs 7 and 8: commit 8060419 (30 Oct 2026, 03:48 IST)

### Runs 7 and 8: pre-registered test plan (written 03:49, before any test eval)

- **Headline policy: run 8 @ 20k** (`experiments/privileged/run8/train/
  act_cpu_v8_abs_dart/checkpoints/020000`), chosen on dev seeds 10500–10549 before
  testing by a rule fixed in advance. **Run 7 @ 6k** (`experiments/privileged/run7/train/
  act_cpu_v7_abs/checkpoints/006000`) is reported alongside as the minimal change (run 2
  with absolute targets).
- **Test seeds: 10700–10799, 100 nominal + 100 shifted, the same seeds for both
  policies** (a paired comparison). This replaces the earlier ledger plan (10700 for run
  7, 10800 for run 8).
- **Reserved:** 10800–10899 for the smallbc official eval; 11000+ untouched for a final
  check.
- **Rule:** each number is reported as it comes, with no re-runs and no checkpoint swaps
  after seeing test results.
- Order: run 8 nominal, run 8 shifted, run 7 nominal, run 7 shifted, all with `sdp eval`
  under heavy+build `flock -o`, `nice -n 10`, after the 5 GB memory wait.
## smallbc: commit 0d9b50d (30 Oct 2026, 06:13 IST)

### smallbc: pre-registered test plan (written before any smallbc test eval)

- **Policy: smallbc @ 25k** (`experiments/smallbc/runs/abs1000/ckpt_025000.pt`). The
  checkpoint rule was fixed in `run_full.sh` before training: most nominal successes on
  dev seeds 10500–10549, ties to the later checkpoint. Dev nominal: 25k 49/50, 40k
  48/50, 50k (last) 48/50 → 25k. (25k dev shifted, run afterwards: 35/50, 15 collision.)
  Files: `experiments/smallbc/results/eval_abs1000_*_s10500_n50.{csv,json}`.
- **Headline:** smallbc @ 25k becomes the README headline policy, decided now on dev
  results only. ACT run 8 @ 20k stays as the ACT result; run 7 and runs 1–6 as they are.
- **Official test:** seeds **10800–10899**, 100 nominal + 100 shifted (the range reserved
  for smallbc in session 7).
- **Paired comparison with ACT:** also seeds **10700–10799**, nominal and shifted (fresh
  for smallbc; runs 7 and 8 have their official results there).
- **Rules:** no re-runs, no checkpoint swaps, every number reported as it comes; seeds
  11000+ stay untouched.
- Commands (each under heavy+build `flock -o`, `nice -n 10`, after the 5 GB wait):
  `python experiments/smallbc/eval.py --ckpt experiments/smallbc/runs/abs1000/ckpt_025000.pt
  --seed-start {10800,10700} --n 100 --distribution {nominal,shifted}
  --condition smallbc_abs1000_25k --out results/eval_smallbc_abs1000_25k_s{seed}_{dist}`.
