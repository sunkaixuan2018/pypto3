# Task Record

## Current Task

- Summary: Validate issue #1503 route/xnorm TensorMap fan-in repro on a2a3 and collect evidence to localize the runtime visibility failure.
- Status: pending_continue
- PR: None
- Issue: <https://github.com/hw-native-sys/pypto/issues/1503>
- Local Worktree: this repository root
- Local Branch: `codex/fix-issue-1503`
- Remote Host: `myserver`
- Remote Repo Path: `/data/sunkaixuan/all_pyptos/codex-fix-issue-1503`
- Fallback Reference Worktree: None
- Latest Tested Commit: unknown
- Last Updated: 2026-05-29 12:03:11 +08:00

## Timer/Skill Context

- Execution Skill: `remote-main-worktree-loop`
- Companion Skills: `skx-remote-workspace-layout` for myserver `/data/sunkaixuan` layout; `pypto-remote-validate` for PyPTO remote validation
- Timer Prompt: `timer_prompt.md`

## Execution Requirements

- Command or target: sync the pushed `codex/fix-issue-1503` branch to `/data/sunkaixuan/all_pyptos/codex-fix-issue-1503`, install dev dependencies, build the a2a3 runtime if needed, then run `python3 -m pytest tests/st/runtime/test_route_xnorm_fanin.py --platform=a2a3 --device={} --save-kernels --dump-passes --enable-dep-gen --enable-l2-swimlane --dump-tensor -v -s` through `task-submit` with the allocated device placeholder preserved.
- Expected result: `test_plain` and `test_side` should pass; `test_chunk_amax` and `test_scale` are expected to reproduce issue #1503 by failing `route_stats` while final `indices` output still passes. If all four pass, preserve artifacts and report that the issue did not reproduce on the tested commit/runtime.
- Required output: stdout, stderr, exit code, tested commit, exact remote repo path, pytest output directory, saved kernel directory, pass dump paths, dep_gen output paths, L2 swimlane JSON paths, tensor dump paths, and any runtime/device logs that identify producer/consumer ordering or GM visibility behavior.
- Verification scope: latest pushed commit on `origin/codex/fix-issue-1503`, validated in the remote worktree on `myserver`.
- Remote repo setup: if `/data/sunkaixuan/all_pyptos/codex-fix-issue-1503` does not exist on `myserver`, create the parent directory and clone the repository there before syncing or validating.
- Workspace layout: for `myserver` paths under `/data/sunkaixuan`, follow `skx-remote-workspace-layout`; put generated `.sh` and ad hoc `.py` helper scripts under `/data/sunkaixuan/codex_sh/`, ad hoc logs and free-form outputs under `/data/sunkaixuan/skx_log_output/`, PyPTO repos under `/data/sunkaixuan/all_pyptos/`, and reuse `/data/sunkaixuan/pypto-lib`.

## Work Completed

- Added `tests/st/runtime/test_route_xnorm_fanin.py` as a four-variant ST repro for issue #1503.
- Confirmed upstream issue #1503 describes the same minimal structure: flat-view route table producer, row-reduction side producer, and base-view scalar-read consumer.
- Ran local syntax verification with `python -m py_compile tests\st\runtime\test_route_xnorm_fanin.py`.

## Current Problem

- Local Windows environment cannot run the ST/codegen path because the built `pypto.pypto_core` extension is unavailable; a2a3 remote validation is required.

## Next Suggested Action

- Commit and push the local repro and timer task files to `origin/codex/fix-issue-1503`.
- On `myserver`, sync or clone `/data/sunkaixuan/all_pyptos/codex-fix-issue-1503` to the pushed branch.
- Run the target a2a3 pytest with dep_gen, L2 swimlane, tensor dump, saved kernels, and pass dumps enabled.
- Inspect whether the final consumer has both ProducerA->ConsumerC and ProducerB->ConsumerC TensorMap edges, and whether the failure points at runtime wait/ready wiring or GM store visibility.

## Task-Specific Notes

- Cross-check status before this task record: Codex baseline and OpenCode-MiMo both favored a runtime/GM-visibility investigation over a missing dep_gen edge; Aider-DeepSeek broadly agreed but its replay details were less reliable; Claude CLI timed out.
- `origin` is the fork `https://github.com/sunkaixuan2018/pypto3.git`; upstream issue source is `https://github.com/hw-native-sys/pypto/issues/1503`.
- The reproduction is intentionally not a runtime fix. The current branch adds focused evidence so the failing condition can be validated and localized on hardware.

## Conclusions

1. Created task record for timer-driven a2a3 validation of the issue #1503 fan-in repro.

## Consolidated Summaries

- None yet.
