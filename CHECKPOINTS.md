# CHECKPOINTS — sim-demo-to-policy

Overnight mode (29 Oct 2026): each checkpoint is logged here, committed, and work continues.

## Plumbing checkpoint — 29 Oct 2026 03:10 IST

Wall-clock so far: 28 min (session started 02:42 IST).

`git status` at the time of the report: clean apart from the uncommitted env work
in progress (`sdp/envs/`, `sdp/expert/`).

| Item | Status | Evidence |
|---|---|---|
| `docker build .` succeeds locally | **N/A on this laptop** | Docker is not installed (MACHINE.md; session addendum forbids installing it). The image is built by CI instead: step "Build image" in run 40772172057 succeeded. |
| CI green on the skeleton | **PASS** | <https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/runs/40772172057> — `completed success … 4m52s`. Log: `MUJOCO_GL=egl: ok`, `MUJOCO_GL=osmesa: ok`, `All checks passed!`, `2 passed in 0.01s`. |
| `pre-commit run --all-files` passes | **PASS** | end-of-file, trailing-whitespace, large-files, ruff check, ruff format, yamllint, markdownlint-cli2 all `Passed`. |
| LICENSE has my name; required files exist | **PASS** | `grep -c "Giridhar J Shenoy" LICENSE` → 1; `ls` lists THIRD_PARTY.md NOTES.md CHANGELOG.md docs/architecture.md docs/results.md scripts/regenerate_all.sh. |
| Conventional commits, none before today | **PASS** | `git log --date=iso`: six commits, all `2026-10-29 02:57:46 +0530`, prefixes chore/ci/docs. |

Not done (recorded in MORNING.md → WAITING FOR GIRI): repo topics (`gh repo edit
--add-topic` was denied by the permission layer), branch protection on `main`.

Does Sunday 1 Nov hold? **Yes** — the skeleton took 28 min; the MUST list is on track.
