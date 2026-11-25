# Simple agent

The course calls `Agent(dispatcher).run()` in `agentic_loop.py`.
Start reading at `solve()`: one conversation reads, edits, builds, submits and
uses grading feedback to repair the same checkout.

- `agentic_loop.py`: prompt, tool schemas, tool execution, model calls and workers.
- `config.py`: selected model, worker count, per-call tokens, time and dollar limits.
- `cost_budget.py`: shared, thread-safe course-token cost accounting, unchanged.
- `sandbox.py` / `sandbox_server.py`: original course repository transport, unchanged.

Tools: `bash` is the general repository tool for discovery, reading, editing and
focused checks; `update_notes` appends durable findings; `build` validates the
current patch; `submit` requests course grading. Build and submit remain explicit
course-workflow controls rather than shell commands.
There are no inspection phases, forced edits, specialist agents or Expert reviews.
Notes are incremental: update_notes appends only the turn's new useful findings,
decisions, edits, scenario traces or check results, tagged with its model-call number.
There is no entry needed for an unchanged turn and no full-summary rewrite.
notes.txt retains every entry, and notes_entry_max_chars caps each entry.
There is no new runtime notes gate. Reads may contain multiple files/ranges,
without a reason gate.
The full solver conversation, including prior tool calls and results, is retained
throughout the task; solver messages are not automatically trimmed. After failed
grading, a separate summary call still prepares the timeline for Expert review.

`build` records a successful command against the exact cumulative patch. Any edit
requires a new build before submission. The model must choose an appropriate build:
a successful command is not proof of behavior or full build coverage. Scenarios
are source-traced; local tests are not generated or run.

Only an explicit `submit` grades a candidate. Truncated model responses never
execute tools. At shutdown, only the last graded patch is retained for the course
dispatcher; unfinished edits are not automatically graded. There is no best-patch
restoration: the course counts the last evaluation.

Set `CONFIG.solver_model` to starter, standard or expert; there is no automatic
escalation. Default workers: 5. Default output caps: Starter 8192, Standard/Expert
16384 tokens per call (subject to available budget). Run budget: $10; task: $0.50.
The course URL, key and selected model ID are read from environment variables.

Reviewer settings in config.py and EXPERT_REVIEWER_PROMPT in agentic_loop.py are
prepared but NOT integrated into solve(). reviewer_enabled defaults to False;
changing it alone currently does nothing. No review calls, review context collector
or additional budget spending are introduced. Future integration must provide the
task, patch, notes, grading history and versioned source evidence; explicitly mark
omitted/stale source rather than claiming the reviewer saw every file.

No course task has been run to validate this new baseline. The old `tests/` folder
is unchanged and targets the retired workflow, not this simplified interface.
