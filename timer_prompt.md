# Timer Prompt

Use `remote-main-worktree-loop`.
Use `skx-remote-workspace-layout` for myserver `/data/sunkaixuan` workspace layout rules.
Use `pypto-remote-validate` for PyPTO-specific validation rules.

Task instance:

- Main worktree: the current repository root
- Task record: `task.md`
- Expected branch: `codex/fix-issue-1503`
- Remote host alias: `myserver`
- Default remote repo path: `/data/sunkaixuan/all_pyptos/codex-fix-issue-1503`
- Missing-file fallback root: None
- Execution depth: bounded-persistent

Task goal:

- Continue the current task recorded in `task.md`
- Prioritize the `Next Suggested Action`
- Keep `task.md` concise, resumable, and reviewable

Required startup reads:

- `task.md`; if it does not exist, pause this run
- `AGENTS.md` if present
- `.claude/CLAUDE.md` if present
- Relevant files under `.claude/rules/` if present
- Companion skills named above
- If required files are missing in the main worktree, use the fallback root when provided

Task-state policy:

- Treat `task.md` status as the only durable task-state source
- If status is `completed`, exit with a brief "task already completed" message
- If status is `in_progress`, exit with a brief "another session is already working" message
- If status is `pause_recommended`, exit with a brief "task is paused for human analysis" message
- Continue only when status is `pending_continue`

Main-worktree policy:

- Treat the designated main worktree as the only code-modification root
- Do not treat the automation worktree as the main repo
- Anchor `task.md`, git inspection, code changes, skill/rule reads, and artifacts to the designated main worktree

Remote sync policy:

- Remote servers can connect to GitHub directly through the configured proxy
- Prefer normal remote git sync with `git fetch`, `git checkout`, and `git pull` in the remote repo
- Do not use git bundle, raw source-copy, or `scp` fallback by default
- If direct remote git sync unexpectedly fails, record the failure in `task.md` and pause or ask for human direction before using a fallback
- For new tasks, default the remote host to `myserver` when no host is specified
- For new tasks, default the remote repo path to `/data/sunkaixuan/all_pyptos/<branch-or-task-dir>` when no path is specified; derive `<branch-or-task-dir>` from the expected branch when possible, such as `/data/sunkaixuan/all_pyptos/auto-deps` for branch `auto-deps`
- If the remote repo path does not exist, create the parent directory and clone the repository into that path before syncing or validating, then record the setup in `task.md`

Workspace layout policy:

- For `myserver` paths under `/data/sunkaixuan`, follow `skx-remote-workspace-layout`
- Put generated `.sh` and ad hoc `.py` helper scripts under `/data/sunkaixuan/codex_sh/`
- Put PyPTO repositories under `/data/sunkaixuan/all_pyptos/`
- Reuse `/data/sunkaixuan/pypto-lib` for pypto-lib instead of creating extra pypto-lib clones
- Put ad hoc logs, free-form outputs, copied artifacts, and temporary bundles under `/data/sunkaixuan/skx_log_output/` unless a command requires a specific output location

Pre-action requirements:

- Before any technical action, update `task.md` to `Status: in_progress` and refresh the timestamp
- Review `Current Task`, `Work Completed`, `Current Problem`, `Next Suggested Action`, recent `Conclusions`, and `Consolidated Summaries`

Unit definitions:

- Treat an action as one small concrete operation, such as reading a log, editing one focused area, syncing a remote repo, or running one command
- Treat an iteration as one resumable handoff unit from `task.md`: identify the current remaining task, make the relevant code change or environment fix, run the appropriate test or validation when safe, then update `task.md` with progress and the next task for a later invocation
- Treat the task as the full goal recorded in `task.md`, which may require many iterations across timer runs

Execution-depth policy:

- For `single-step`, perform only the first clear action, then update `task.md`
- For `bounded-persistent`, complete 1-2 bounded iterations around the current `Next Suggested Action`; across the whole invocation, do at most 1-2 fix or progress attempts and at most 1-2 validation or rerun attempts
- For `extended-bounded`, complete 3-5 bounded iterations around the current remaining problem or one larger phase; use it only when explicitly selected in the task instance or user request
- For `exhaustive-until-blocked`, keep advancing while safe actions remain
- Stop after the `bounded-persistent` or `extended-bounded` iteration budget is used, even if more safe actions remain
- If work remains after a bounded budget is used, set `task.md` to `pending_continue`, record remaining work in `Next Suggested Action`, and end the run
- Stop for destructive risk, missing permissions, ambiguous product decisions, dirty-state conflicts, repeated low-signal attempts, or resource/time concerns

End-of-run requirements:

- Update `task.md` before ending and never leave it as `in_progress`
- Set `completed`, `pending_continue`, or `pause_recommended`
- Keep `Work Completed` durable, `Current Problem` singular, and `Next Suggested Action` limited to 2-4 actions
- Append exactly one short conclusion for this run

Output requirements:

- Output only a brief conclusion and next-step suggestion in Chinese
- State whether meaningful progress was made, the current main blocker, and whether code was modified, committed, pushed, or synced
- Do not add task-specific result criteria here; read and report those from `task.md`
