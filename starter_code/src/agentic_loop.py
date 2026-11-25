"""Simple course agent: one conversation, ordinary tools, explicit grading.

The course entry point remains Agent(dispatcher).run(). Start reading at solve().
All repository access goes through Sandbox; only traces live in this process.
"""

import hashlib
import json
import os
import posixpath
import re
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dispatcher import DispatcherError, DispatcherShutdown
from .config import CONFIG
from .cost_budget import BudgetExceeded, CostBudget, RATES, field, input_sample, measured_input_bound, token_cost
from .sandbox import Sandbox


SYSTEM_PROMPT = """
Task_description:
{problem_statement}

TASK_requirements:
{requirements}

Task_Interface:
{interface}

You are a software engineer with task requirements given to you. You need to
understand the requirement by analysing the Task_description, TASK_requirements,
Task_Interface. These are high level requirements and not literal code. You need
to search the codebase and files to analyse and make a plan to implement the
task based on the task specs such as Task_description, TASK_requirements and
Task_Interface. Every requirement is important, the hidden tests evaluation of
code implementation will fail if you miss even a single task requirement or spec.
As soon as you finish the implementation, build the code and then immedeatly
submit.

You need to make sure to not fall into inspection for a long time. Our main goal
is to implement and submit for grading. Grading evaluates your implemented code
based on hidden tests which we do not know anything about. Task is considered
finished only if every single hidden test passes. Please check how many tests
failed in all previous grading attempts to get better. To get better try to
evaluate Task_description, each TASK_requirements, Task_Interface against the patch
and think what are we missing? did we miss a corner case or behaviour? and
implement for next round. Make sure to submit otherwise we don't get any points.

Keep internal reasoning under 150 words.

To avoid reading/searching files over and over, please make notes(update_notes
tool) and utilize in future to avaid already searched files.
Investigation is a short evidence loop, not an exhaustive search:
1. Identify the specific unknown that blocks diagnosis or implementation.
2. Make one focused discovery batch (list/search/read only what can answer it).
3. Reassess using the returned evidence and prior notes. If the affected symbol,
   required behavior, and mismatch are clear, implement the smallest complete fix
   in the next action; do not search for more examples first.
4. If still blocked, name ONE precise unanswered question and make ONE targeted
   read/search to answer it, then reassess. Do not repeat a query that returned no
   matches; change the query only when you have a new hypothesis or scope.
After Bash, use its workspace_state to see whether the checkout changed. If it
shows a source change, inspect that diff, build, and submit rather than rereading
unchanged areas. If repeated Bash actions leave the checkout unchanged, stop
searching and act on the evidence already collected or name the precise blocker.
"""

# Independent review runs after failed grading.
EXPERT_REVIEWER_PROMPT = """You are an experienced software engineer independently
reviewing a coding agent's patch after failed grading. Identify what the solver
missed using EVIDENCE, not confidence, speculation or a restatement of its plan.

Task_description:
{problem_statement}

TASK_requirements:
{requirements}

Task_Interface:
{interface}

Solver notes (incremental history, unverified claims; later corrections supersede earlier claims):
{notes}

Implementation patch:
{patch}

Grading results and most recent prior Expert review, if any:
{grading_history}

Requirement-by-requirement audit: Before reporting findings, check every stated task
requirement and interface contract.
For each, trace the relevant behavior in the patch and classify it as satisfied,
violated, or uncertain, citing concrete code behavior. Assess the implementation against
the requirements individually and in combination. Treat notes and prior reviews as
fallible; verify them against the task details and current patch. Prioritize concrete
correctness gaps relevant to the grading result. Be concise and do not claim to know
hidden test cases.

State concise 1 liner for each requirement "how the patch works or doesn't work?".
Create finding for each "patch doesn't satisfy" scenario.
"""

def _debug(message):
    print(f"[madsOpt] {message}", file=sys.stderr, flush=True)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def function(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required)}}}


TOOLS = [
    function("bash", "General repository tool, like a task-container shell: discover files, "
             "search/read source, edit files, and run focused checks. Each command starts in the "
             "task checkout. Use targeted commands such as rg --files, rg -n, and sed; inspect "
             "results and avoid repeated broad searches. Shell runs with errexit and pipefail, so "
             "handle expected no-match statuses (rg exit 1) explicitly and avoid grep|head pipelines. "
             "Results include a concise working-tree status and fingerprint; use it to confirm whether "
             "your action changed the candidate. "
             "Use the dedicated build action for the final build and submit action for grading. "
             "Example: {\"command\":\"rg -n 'target_symbol' src\"}.", {
        "command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"]),
    function("update_notes", "Append a concise entry containing ONLY this turn's new findings, "
             "plan changes, edits, concrete scenario traces, build/grading results or unresolved "
             "questions. Include inspected paths/ranges where relevant. Required when useful new "
             "information exists; skip unchanged turns. Never rewrite the notebook or copy files. "
             "Explicitly correct superseded findings; do not claim unrun tests passed. "
             "Example: {\"notes\":\"Read src/main.py:10-30; caller passes a dictionary.\"}.", {
        "notes": {"type": "string"}}, ["notes"]),
    function("build", "After the final source edit, run the narrowest valid build/typecheck/compile "
             "command here (not through Bash and not tests). Any later source edit requires another "
             "build call. Only a successful build of the current patch satisfies submit. If submit says "
             "the patch is not built, call this tool next. "
             "Example: {\"command\":\"python3 -m py_compile src/main.py\"}.", {
        "command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"]),
    function("submit", "Call alone to grade the complete cumulative patch, only after the dedicated "
             "build tool has succeeded on the current patch. Bash builds do not satisfy the gate. "
             "Returns failing/total test counts for further repair. Example: {}.", {}),
]


SEARCH_SCRIPT = r'''
import fnmatch,json,re,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); args=json.loads(sys.argv[2])
mode=args['mode']; pattern=args['pattern']; limit=args['limit']; cap=args['cap']
try:
    rx=re.compile(pattern) if args.get('regex') else None
except re.error as exc:
    print(json.dumps({'error':'Invalid Python regex: '+str(exc),
                      'hint':'Use regex=false for literal punctuation.'})); sys.exit(0)
listed=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                      cwd=root,capture_output=True)
if listed.returncode:
    print(json.dumps({'error':'File discovery failed: '+listed.stderr.decode(errors='replace')})); sys.exit(0)
scopes=[]
for path in args.get('paths',['.']):
    resolved=(root/path).resolve()
    if not resolved.is_relative_to(root):
        print(json.dumps({'error':'Search path is outside checkout: '+path})); sys.exit(0)
    if not resolved.exists():
        print(json.dumps({'error':'Search path not found: '+path,'hint':'Use list_files to discover the path.'})); sys.exit(0)
    scopes.append(resolved)
items=[]; used=0; truncated=False; warnings=[]
for name in sorted(set(listed.stdout.decode(errors='replace').split('\0'))):
    if not name: continue
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file(): continue
    if not any(path==scope or path.is_relative_to(scope) for scope in scopes): continue
    if mode=='list_files':
        if not (fnmatch.fnmatchcase(name,pattern) or
                (pattern.startswith('**/') and fnmatch.fnmatchcase(name,pattern[3:]))): continue
        rows=[name]
    else:
        try:
            if path.stat().st_size>10000000:
                warnings.append(name+': skipped file over 10MB'); continue
            with path.open('rb') as stream:
                if b'\0' in stream.read(4096): continue
            def matching_rows():
                with path.open(encoding='utf-8',errors='replace') as stream:
                    for number,line in enumerate(stream,1):
                        if (rx.search(line) if rx else pattern in line):
                            value={'path':name,'line':number,'text':line.rstrip('\r\n')[:1000]}
                            if len(line.rstrip('\r\n'))>1000: value['line_truncated']=True
                            yield value
            rows=matching_rows()
        except OSError as exc:
            warnings.append(name+': '+str(exc)); continue
    try:
        for item in rows:
            size=len(json.dumps(item,ensure_ascii=False))+2
            if len(items)>=limit or used+size>cap:
                truncated=True; break
            items.append(item); used+=size
    except OSError as exc:
        warnings.append(name+': '+str(exc))
    if truncated: break
print(json.dumps({'files' if mode=='list_files' else 'matches':items,'truncated':truncated,
                  'warnings':warnings[:10]},ensure_ascii=False))
'''


def call_model(client, model_id, messages, logger, iteration, **kwargs):
    """Course API request; every completed response is recorded in the course trace."""
    response = client.chat.completions.create(model=model_id, messages=messages, **kwargs)
    usage = response.usage
    logger.api_request(iteration, field(usage, "prompt_tokens"), field(usage, "completion_tokens"),
                       field(usage, "total_tokens"), model_id)
    return response


class Agent:
    def __init__(self, dispatcher, config=None, client=None, sandbox_factory=Sandbox):
        self.dispatcher = dispatcher
        self.config = config or CONFIG
        cfg = self.config
        self.model_id = os.environ.get(f"CS2680_MODEL_{cfg.solver_model.upper()}", cfg.solver_model)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=os.environ["CS2680_BASE_URL"],
                            api_key=os.environ["CS2680_API_KEY"], max_retries=0)
        self.client = client
        self.sandbox_factory = sandbox_factory
        self.api_slots = threading.BoundedSemaphore(cfg.api_concurrency)
        # Solver and reviewer share the same per-task/run budgets and call allowance.
        self.budget = CostBudget(cfg.run_budget_usd, cfg.task_budget_usd,
                                 cfg.task_budget_usd, cfg.max_model_calls, cfg.max_model_calls)

    def run(self):
        """One worker owns each task until done; the dispatcher permits five live tasks."""
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            futures = [pool.submit(self.worker) for _ in range(self.config.workers)]
            for future in futures:
                future.result()

    def worker(self):
        while True:
            try:
                task = self.dispatcher.next_task()
            except DispatcherShutdown:
                return
            except DispatcherError as exc:
                if "live" not in str(exc).lower():
                    raise
                time.sleep(1)
                continue
            if task is None:
                return
            try:
                self.solve(task)
            except DispatcherShutdown:
                return

    def patch(self, key):
        """extract_patch returns a byte count, not the diff text."""
        self.dispatcher.extract_patch(key)
        return Path(self.dispatcher.patch_path(key)).read_text()

    def command(self, sb, arguments, remaining, tool_name="bash"):
        command = arguments["command"]
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty text")
        # Catch common accidental device deletion/output replacement. This is an
        # agent-side safety check, not a security boundary for arbitrary shell code.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';|&()<>')
        lexer.whitespace_split = True
        words = list(lexer)
        for index, word in enumerate(words):
            output = None
            if word in ('-o', '--output', '--output-file') and index + 1 < len(words):
                output = words[index + 1]
            elif word.startswith(('--output=', '--output-file=')):
                output = word.split('=', 1)[1]
            elif word.startswith('-o') and len(word) > 2:
                output = word[2:]
            if output and posixpath.normpath(output) == '/dev/null':
                raise ValueError("Do not use /dev/null as an output-file argument: tools may unlink it. "
                                 "Use a temporary regular file instead. Shell redirection to /dev/null is allowed.")
        if '/dev/null' in [posixpath.normpath(word) for word in words] and any(
                posixpath.basename(word) in ('rm', 'unlink', 'mv', 'install') for word in words):
            raise ValueError("Command may remove or replace /dev/null; preserve sandbox system devices")
        timeout = max(1, min(900, int(arguments.get("timeout_s", 60)), int(remaining)))
        started = time.monotonic()
        result = sb.exec("bash -e -o pipefail -c " + shlex.quote(command), timeout_s=timeout)
        result["duration_s"] = round(time.monotonic() - started, 3)
        output = result.get("output", "")
        cap = self.config.tool_output_chars
        if len(output) > cap:
            result["output"] = output[:cap // 2] + "\n[OUTPUT OMITTED]\n" + output[-cap // 2:]
            result["output_truncated"] = True
        if not output.strip():
            result["output"] = "[command produced no output]"
        if tool_name == "bash":
            # Show whether the command changed the candidate. This is a concise
            # feedback signal, not a replacement for inspecting the diff.
            state_command = (
                "status=$(git status --short --untracked-files=all 2>&1) || exit $?; "
                "stat=$(git diff --stat HEAD 2>&1) || exit $?; "
                "tracked=$(git diff --binary HEAD | git hash-object --stdin); "
                "untracked=$(git ls-files --others --exclude-standard -z | "
                "while IFS= read -r -d '' path; do printf '%s ' \"$path\"; "
                "git hash-object -- \"$path\"; done | sort | git hash-object --stdin); "
                "printf 'Status:\\n%s\\nDiff summary:\\n%s\\nTracked fingerprint: %s\\nUntracked fingerprint: %s\\n' "
                "\"$status\" \"$stat\" \"$tracked\" \"$untracked\""
            )
            try:
                state = sb.exec("bash -e -o pipefail -c " + shlex.quote(state_command), timeout_s=10)
                if state.get("exit_code") == 0 and not state.get("timed_out"):
                    state_output = state.get("output", "").strip()
                    result["workspace_state"] = state_output or "No repository changes detected."
                else:
                    result["workspace_state"] = "Working-tree status unavailable."
            except Exception as exc:
                result["workspace_state"] = f"Working-tree status unavailable ({type(exc).__name__})."
        if result.get("exit_code", 0) != 0 and not result.get("timed_out"):
            if tool_name == "build":
                result["hint"] = "Build failed; inspect the compiler output, fix the reported issue, then rebuild before submitting."
            else:
                result["hint"] = ("Command exited nonzero; inspect the command and exit status before retrying. "
                                  "If it was grep/rg, exit code 1 means no matches. Refine the command or "
                                  "narrow its scope instead of repeating it unchanged.")
        return result

    def read_files(self, sb, arguments):
        files, pending, room = [], [], self.config.tool_output_chars
        requests = arguments.get("requests", arguments.get("files"))
        if requests is None:
            for k, v in arguments.items():
                if "requests" in k or "files" in k:
                    if "[" in k and "]" in k:
                        m = re.search(r'\[.*\]', k, re.DOTALL)
                        if m:
                            try:
                                requests = json.loads(m.group(0))
                                break
                            except Exception:
                                pass
                    requests = v
                    break
            else:
                requests = [arguments]
        if isinstance(requests, str):
            try:
                decoded = json.loads(requests)
                if isinstance(decoded, (list, dict)):
                    requests = decoded
            except ValueError:
                pass
        if not isinstance(requests, list):
            requests = [requests] if requests else []
        if not requests:
            raise ValueError("requests must be a nonempty list")
        for index, request in enumerate(requests):
            if room < 200:
                pending = requests[index:]
                break
            if isinstance(request, str):
                request = {"path": request}
            if not isinstance(request, dict):
                files.append({"request": request, "error": "Each request must be an object with path; "
                              "example: {\"requests\":[{\"path\":\"src/main.py\"}]}"})
                continue
            path = request.get("path")
            if not isinstance(path, str) or not path.strip():
                files.append({"request": request, "error": "Missing nonempty path in read request; "
                              "example: {\"requests\":[{\"path\":\"src/main.py\"}]}"})
                continue
            try:
                lines = sb.read_text(path).splitlines()
                start = max(1, int(request.get("start_line", 1) or 1))
                count = min(self.config.read_max_lines,
                            max(1, int(request.get("num_lines", self.config.read_default_lines) or self.config.read_default_lines)))
                rows, next_line = [], None
                for number in range(start, min(len(lines) + 1, start + count)):
                    row = f"{number}: {lines[number - 1]}"
                    if len(row) + 1 > room:
                        next_line = number
                        break
                    rows.append(row)
                    room -= len(row) + 1
                end = start + len(rows) - 1
                if next_line is None and end < len(lines):
                    next_line = max(start, end + 1)
                files.append({"path": path, "content": "\n".join(rows), "total_lines": len(lines),
                              "next_start_line": next_line})
            except Exception as exc:
                files.append({"path": path, "error": f"{type(exc).__name__}: {exc}",
                              "hint": "Use list_files for missing paths; line offsets must be integers."})
        return {"files": files, "pending_requests": pending,
                "note": "For an oversized single line use targeted bash extraction."}

    @staticmethod
    def replace_text(sb, arguments):
        edits = arguments.get("edits")
        if edits is None:
            for k, v in arguments.items():
                if "edits" in k:
                    edits = v
                    break
            else:
                if all(k in arguments for k in ("path", "old_text", "new_text")):
                    edits = [arguments]
        if isinstance(edits, str):
            try:
                decoded = json.loads(edits)
                if isinstance(decoded, (list, dict)):
                    edits = decoded
            except ValueError:
                pass
        if not isinstance(edits, list):
            edits = [edits] if edits else []
        if not edits:
            raise ValueError("edits must be a nonempty list")
        pending = {}
        for index, edit in enumerate(edits):
            example = '{"edits":[{"path":"src/main.py","old_text":"x = 1","new_text":"x = 2"}]}'
            if not isinstance(edit, dict):
                raise ValueError(f"Edit {index + 1} must be an object. Example: {example}")
            path = edit.get("path", arguments.get("path"))
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Edit {index + 1} needs path inside the edit or at top level. Example: {example}")
            if "old_text" not in edit or "new_text" not in edit:
                raise ValueError(f"Edit {index + 1} needs old_text and new_text. Example: {example}")
            old, new = edit["old_text"], edit["new_text"]
            if not all(isinstance(value, str) for value in (path, old, new)):
                raise ValueError("path, old_text and new_text must be strings")
            if not old or old == new:
                raise ValueError("old_text must be nonempty and different from new_text")
            text = pending[path] if path in pending else sb.read_text(path)
            if text.count(old) != 1:
                raise ValueError(f"old_text must match exactly once in {path}; read fresh source")
            pending[path] = text.replace(old, new, 1)
        for path, text in pending.items():
            sb.write_text(path, text)
        return {"edited_files": list(pending)}

    def search_repository(self, sb, name, arguments, remaining):
        """One shell-safe implementation for literal/regex search and glob discovery."""
        pattern = arguments.get('pattern')
        if not isinstance(pattern, str) or not pattern:
            raise ValueError('pattern must be nonempty text')
        paths = arguments.get('paths', ['.'])
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError('paths must be a nonempty list of file/directory names')
        regex = arguments.get('regex', False)
        if type(regex) is not bool:
            raise ValueError('regex must be true or false; use false for literal matching')
        args = {'mode': name, 'pattern': pattern, 'paths': paths, 'regex': regex,
                'limit': max(1, min(500, int(arguments.get('limit', 100)))),
                'cap': self.config.tool_output_chars}
        command = 'python3 -c ' + shlex.quote(SEARCH_SCRIPT) + ' ' + shlex.quote(sb.workdir) + ' ' + shlex.quote(json.dumps(args))
        result = sb.exec(command, timeout_s=max(1, min(60, int(remaining))))
        if result.get('exit_code') or result.get('timed_out'):
            return {'error': result.get('output') or 'Search failed or timed out'}
        payload = json.loads(result['output'])
        if payload.get('error'):
            return payload
        result_key = 'files' if name == 'list_files' else 'matches'
        if not payload.get(result_key):
            payload['message'] = ('No files matched this glob.' if name == 'list_files'
                                  else 'No source lines matched this literal/regex search.')
        if payload.get('truncated'):
            payload['next_step'] = 'Narrow paths or the pattern, or request a smaller result limit.'
        return payload

    @staticmethod
    def write_files(sb, arguments):
        """Write a validated batch and report any partial completion precisely."""
        files = arguments.get('files')
        if not isinstance(files, list) or not files:
            raise ValueError('files must be a nonempty list of {path, content} objects')
        validated, seen = [], set()
        for index, item in enumerate(files, 1):
            if not isinstance(item, dict):
                raise ValueError(f'File {index} must be an object with path and content')
            path, content = item.get('path'), item.get('content')
            if not isinstance(path, str) or not path.strip() or path.startswith('/'):
                raise ValueError(f'File {index} path must be a nonempty repository-relative path')
            normalized = posixpath.normpath(path)
            if normalized == '..' or normalized.startswith('../'):
                raise ValueError(f'File {index} path must stay inside the repository')
            if normalized in seen:
                raise ValueError(f'Duplicate path in write batch: {normalized}')
            if not isinstance(content, str):
                raise ValueError(f'File {index} content must be a string')
            seen.add(normalized)
            validated.append((normalized, content))

        written, failures = [], []
        for path, content in validated:
            try:
                existed = sb.write_text(path, content)
                written.append({'path': path, 'result': 'replaced' if existed else 'created'})
            except Exception as exc:
                failures.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})
                break
        return {'written': written, 'written_files': [item['path'] for item in written],
                'failures': failures,
                'partial': bool(failures and written),
                'message': ('Some files were written before an error; do not repeat the whole batch blindly.'
                            if failures and written else
                            'One or more files could not be written.' if failures else
                            'All requested files were written.')}

    def append_note(self, directory, text, call):
        """Persist one turn entry without overwriting earlier entries."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("notes must be nonempty text")
        if len(text) > self.config.notes_entry_max_chars:
            raise ValueError(f"Send only this turn's new information, under {self.config.notes_entry_max_chars} characters")
        entry = f"[Turn {call}]\n{text.strip()}\n\n"
        with (directory / "notes.txt").open("a") as stream:
            stream.write(entry)
        return entry

    @staticmethod
    def solver_evidence_context(observations, changed_paths=(), max_chars=24000):
        """Compact durable source/tool evidence into bounded, categorized JSON."""
        limit = max(256, int(max_chars))
        context = {
            "repository_paths": [],
            "searches": [], "read_snippets": [], "edited_files": [],
            "omitted_evidence": {"searches": 0, "reads": 0, "edits": 0, "commands": 0},
            "warning": "Captured evidence can be stale after edits; refresh source before relying on it. A no-match is not proof that behavior is absent.",
        }

        def encoded(candidate):
            return json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))

        def add_path(path):
            if not isinstance(path, str) or not path or path in context["repository_paths"]:
                return
            candidate = dict(context)
            candidate["repository_paths"] = [*context["repository_paths"], path]
            # Keep path inventories from crowding out exact search/read evidence.
            if len(encoded(candidate)) <= min(limit, max(256, limit // 3)):
                context["repository_paths"].append(path)

        def add(section, item, category):
            candidate = dict(context)
            candidate[section] = [*context[section], item]
            if len(encoded(candidate)) <= limit:
                context[section].append(item)
            else:
                context["omitted_evidence"][category] += 1

        initial_paths = sorted(changed_paths) if isinstance(changed_paths, (set, frozenset)) else changed_paths
        for path in initial_paths:
            add_path(path)

        for observation in observations or ():
            tool = observation.get("tool")
            result = observation.get("result") or {}
            if tool == "list_files":
                for path in result.get("files", []):
                    add_path(path)
            elif tool == "search_files":
                matches = result.get("matches", [])
                if matches:
                    add("searches", {"call": observation.get("call"),
                         "pattern": (observation.get("arguments") or {}).get("pattern"),
                         "matches": matches}, "searches")
            elif tool == "read_files":
                for item in result.get("files", []):
                    if not isinstance(item, dict) or item.get("error"):
                        continue
                    content = item.get("content", "")
                    first = re.match(r"\s*(\d+):", content)
                    add("read_snippets", {"call": observation.get("call"),
                         "path": item.get("path"), "start_line": int(first.group(1)) if first else 1,
                         "total_lines": item.get("total_lines"), "content": content}, "reads")
                    path = item.get("path")
                    add_path(path)
            elif tool in ("replace_text", "write_files"):
                edited = result.get("edited_files", result.get("written_files", []))
                if edited:
                    add("edited_files", {"call": observation.get("call"),
                         "edited_files": edited}, "edits")
            elif tool in ("bash", "build", "run_tests"):
                summary = {"call": observation.get("call"), "tool": tool,
                           "command": (observation.get("arguments") or {}).get("command"),
                           "exit_code": result.get("exit_code"), "timed_out": result.get("timed_out"),
                           "output": str(result.get("output", ""))[-1200:]}
                candidate = dict(context)
                candidate.setdefault("commands", []).append(summary)
                if len(encoded(candidate)) <= limit:
                    context.setdefault("commands", []).append(summary)
                else:
                    context["omitted_evidence"]["commands"] += 1

        # Paths are cheap and useful; keep the serialized artifact strictly bounded.
        while len(encoded(context)) > limit and context["repository_paths"]:
            context["repository_paths"].pop()
        result = encoded(context)
        if len(result) <= limit:
            return result
        # A very small caller-supplied cap still gets valid JSON and omission metadata.
        minimal = {"repository_paths": [], "searches": [], "read_snippets": [],
                   "edited_files": [], "commands": [],
                   "omitted_evidence": context["omitted_evidence"],
                   "warning": "Evidence omitted to honor context bound; no-match is not proof."}
        return encoded(minimal)

    def compact(self, messages, observations=(), changed_paths=(), evidence_context=None):
        """Bound solver history while retaining the task header and whole tool exchanges."""
        limit = int(getattr(self.config, "context_max_chars", 120000))
        if len(json.dumps(messages, ensure_ascii=False, default=str)) <= limit:
            return messages
        roots = list(messages[:2])
        evidence_header = "Captured repository evidence (historical; refresh after edits):\n"
        root_size = len(json.dumps(roots, ensure_ascii=False, default=str))
        evidence_room = limit - root_size - len(evidence_header) - 32
        if evidence_room >= 256:
            evidence_limit = min(evidence_room, int(getattr(self.config, "evidence_context_chars", 24000)))
            evidence = evidence_context or ""
            if not evidence or len(evidence) > evidence_limit:
                evidence = self.solver_evidence_context(observations, changed_paths,
                                                        max_chars=evidence_limit)
            evidence_message = {"role": "user", "content": evidence_header + evidence}
        else:
            evidence_message = {"role": "user", "content":
                "Captured repository evidence omitted to preserve the task prompt within the context limit."}
        roots.append(evidence_message)
        if len(json.dumps(roots, ensure_ascii=False, default=str)) > limit:
            return roots[:2]

        # Each assistant response and its following tool/user results form one indivisible
        # exchange. Select from the end so truncation cannot leave a tool result orphaned.
        groups, group = [], []
        for message in messages[2:]:
            if message.get("role") == "assistant" and group:
                groups.append(group)
                group = [message]
            else:
                group.append(message)
        if group:
            groups.append(group)
        kept = []
        recent = max(1, int(getattr(self.config, "context_recent_exchanges", 6)))
        for exchange in reversed(groups[-recent:]):
            candidate = roots + exchange + kept
            if len(json.dumps(candidate, ensure_ascii=False, default=str)) > limit:
                continue
            kept = exchange + kept
        compacted = roots + kept
        return compacted

    def solve(self, task):
        """Read -> implement -> build -> explicit submit -> grading feedback -> repair."""
        cfg, key = self.config, int(task["task"])
        logger = self.dispatcher.logger(key)
        sb = self.sandbox_factory(task["sandbox_url"], task["workdir"])
        directory = Path(cfg.log_dir) / str(key)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        calls, notes, grades, seen = 0, [], [], set()
        last_workspace_state, unchanged_bash_calls, progress_nudge_sent = None, 0, False
        built_patch, last_submitted, sample = None, "", None
        source_paths, observations, reviews = set(), [], []
        latest_review = ""
        reason = "task allowance reached"
        solver_prompt = SYSTEM_PROMPT.format(
            problem_statement=task.get("problem_statement", ""),
            requirements=task.get("requirements", ""),
            interface=task.get("interface", ""))

        messages = [{"role": "system", "content": solver_prompt},
                    {"role": "user", "content": "Implement the task described above."}]

        def remaining():
            return max(0, cfg.task_timeout_s - (time.monotonic() - started))

        def event(name, **data):
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"event": name, "elapsed_s": round(time.monotonic() - started, 3),
                                         **data}, ensure_ascii=False, default=str) + "\n")

        def request(tier, role, conversation, tools, output_tokens, repair_reserve=0):
            """Retry only transport failures and transient HTTP statuses, never tool execution."""
            from openai import APIConnectionError, APIStatusError

            for attempt in range(cfg.api_max_retries + 1):
                try:
                    return request_once(tier, role, conversation, tools, output_tokens, repair_reserve)
                except (APIConnectionError, APIStatusError) as exc:
                    status = getattr(exc, 'status_code', None)
                    transient = isinstance(exc, APIConnectionError) or status in (408, 409, 429) or (
                        status is not None and status >= 500)
                    if not transient or attempt >= cfg.api_max_retries:
                        raise
                    delay = min(2 ** attempt, 8)
                    response = getattr(exc, 'response', None)
                    retry_after = response.headers.get('retry-after') if response is not None else None
                    try:
                        delay = max(delay, float(retry_after)) if retry_after else delay
                    except (ValueError, TypeError):
                        pass
                    reserve_s = 60 if role in ('reviewer', 'summarizer') else 0
                    if remaining() <= delay + reserve_s + 1 or calls >= cfg.max_model_calls:
                        raise
                    event('api_retry', role=role, tier=tier, retry=attempt + 1,
                          delay_s=delay, status=status, error=str(exc))
                    time.sleep(delay)  # request_once released the shared API slot already.

        def request_once(tier, role, conversation, tools, output_tokens, repair_reserve=0):
            """One budgeted/logged API gateway for solver, summarizer and reviewer."""
            nonlocal calls, sample
            if calls >= cfg.max_model_calls or remaining() <= 0:
                raise BudgetExceeded("task time/model-call allowance reached")
            if not self.api_slots.acquire(timeout=max(0.01, remaining())):
                raise BudgetExceeded("deadline waiting for API slot")
            try:
                if remaining() <= (60 if role in ('reviewer', 'summarizer') else 0):
                    raise BudgetExceeded("insufficient time for this call and subsequent repair")
                model = os.environ.get(f"CS2680_MODEL_{tier.upper()}", tier)
                options = {"model": model}
                if tools:
                    options.update(tools=tools, tool_choice="auto")
                bound = self.budget.estimate_input(key, conversation, options, sample if role == 'solver' else None)
                if repair_reserve:
                    budget = self.budget.snapshot(key)
                    available = min(budget['task_remaining_usd'], budget['run_remaining_usd']) - repair_reserve
                    affordable = int((available - token_cost(tier, bound, 0)) * 1_000_000 / RATES[tier][2])
                    output_tokens = min(output_tokens, affordable)
                    if output_tokens < 512:
                        raise BudgetExceeded(f"insufficient review budget: estimated input={bound} tokens "
                            f"(${token_cost(tier, bound, 0):.4f}), available after repair reserve=${available:.4f}")
                ticket = self.budget.reserve(key, tier, role, conversation, options,
                    output_tokens, input_bound=bound, repair_reserve=repair_reserve)
                calls += 1
                event("model_input", call=calls, role=role, tier=tier, messages=conversation,
                      output_limit=ticket.output_limit)
                _debug(f"task {key}: {role}/{tier}, call={calls}")
                try:
                    response = call_model(self.client, model, conversation, logger, calls,
                        max_tokens=ticket.output_limit,
                        timeout=min(cfg.api_timeout_s, max(1, remaining() - (60 if role in ('reviewer', 'summarizer') else 0))),
                        **{name: value for name, value in options.items() if name != 'model'})
                except Exception:
                    event("api_error", role=role, tier=tier, accounting=self.budget.settle(ticket))
                    raise
                accounting = self.budget.settle(ticket, response.usage)
                if role == 'solver':
                    sample = input_sample(conversation, options, response.usage)
                event("api_response", call=calls, role=role, tier=tier,
                      finish_reason=response.choices[0].finish_reason,
                      response=response.model_dump(), accounting=accounting)
                return response.choices[0]
            finally:
                self.api_slots.release()

        def review(patch):
            """One independent review per failed grading; repair can continue on failure."""
            if not cfg.reviewer_enabled:
                return ""
            try:
                if remaining() < cfg.reviewer_min_remaining_s or calls + 2 >= cfg.max_model_calls:
                    raise BudgetExceeded("insufficient remaining time/calls for review and repair")
                fields = dict(
                    problem_statement=task.get('problem_statement', ''),
                    requirements=task.get('requirements', ''), interface=task.get('interface', ''),
                    notes=''.join(notes) or '[No solver notes recorded.]',
                    patch=patch,
                    grading_history=json.dumps({
                        'grades': grades,
                        'most_recent_prior_review': reviews[-1:] or None,
                    }, ensure_ascii=False))
                prompt = EXPERT_REVIEWER_PROMPT.format(**fields)
                if len(prompt) > cfg.reviewer_context_chars:
                    raise BudgetExceeded("review context exceeds configured limit")
                choice = request(cfg.reviewer_model, 'reviewer',
                    [{"role": "system", "content": prompt},
                     {"role": "user", "content": "Review this failed candidate; return evidence-backed findings."}], [],
                    min(cfg.reviewer_output_tokens, getattr(cfg, f'{cfg.reviewer_model}_output_tokens')),
                    repair_reserve=cfg.reviewer_repair_reserve_usd)
                if choice.finish_reason != 'stop' or not choice.message.content or choice.message.tool_calls:
                    raise ValueError("review incomplete/truncated or empty; partial advice discarded")
                advice = choice.message.content
                reviews.append({'grading_attempt': grade['attempt'], 'advice': advice})
                (directory / f'review_{len(grades)}.txt').write_text(advice)
                event('review_completed', attempt=len(grades), model=cfg.reviewer_model, advice=advice)
                return advice
            except DispatcherShutdown:
                raise
            except Exception as exc:
                event('review_skipped', attempt=len(grades), reason=f'{type(exc).__name__}: {exc}')
                return f"Review unavailable ({type(exc).__name__}: {exc}). Continue evidence-based repair yourself."

        try:
            while remaining() > 0 and calls < cfg.max_model_calls:
                turn_status = f"\n\n[Status: Turn {calls + 1}/{cfg.max_model_calls} | Remaining time: {int(remaining())}s | Grading attempts: {len(grades)}/{cfg.max_evaluations}]"
                # Inject accumulated notes into system prompt so solver has persistent memory.
                notes_context = ""
                if notes:
                    notes_cap = cfg.notes_context_chars
                    recent_notes, used_chars = [], 0
                    for note in reversed(notes):
                        if used_chars + len(note) > notes_cap:
                            break
                        recent_notes.append(note)
                        used_chars += len(note)
                    recent_notes.reverse()
                    omitted = len(notes) - len(recent_notes)
                    prefix = f"[{omitted} earlier note entries omitted; captured tool evidence is retained.]\n" if omitted else ""
                    notes_context = ("\n\nYour investigation notes (persistent memory — use these instead of re-reading):\n" +
                                     prefix + ''.join(recent_notes))
                messages[0] = {"role": "system", "content": solver_prompt + notes_context + turn_status}
                evidence_context = self.solver_evidence_context(
                    observations, source_paths, max_chars=cfg.evidence_context_chars)
                messages = self.compact(messages, observations, source_paths, evidence_context)
                if latest_review and not any(latest_review in msg.get('content', '') for msg in messages):
                    messages.append({"role": "user", "content": "Latest independent review (verify claims):\n" + latest_review})
                choice = request(cfg.solver_model, 'solver', messages, TOOLS, cfg.output_tokens)
                if choice.finish_reason == "length":
                    messages.append({"role": "user", "content":
                        "Your last response was truncated; no partial tools were executed. "
                        "Continue with a smaller, complete tool action. Do not restart the task."})
                    continue
                message = choice.message.model_dump(include={"role", "content", "tool_calls"},
                                                    exclude_none=True)
                messages.append(message)
                tool_calls = message.get("tool_calls", [])
                if not tool_calls:
                    messages.append({"role": "user", "content":
                        "Continue using tools to complete the fix. When ready, build and call submit."})
                passed = False
                progress_checkpoint = False
                for call in tool_calls:
                    name = call["function"]["name"]
                    tool_iteration, args = calls, {}
                    try:
                        args = json.loads(call["function"]["arguments"])
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be an object")
                        logger.tool_call(calls, name, args)
                        if name == "read_files":
                            result = self.read_files(sb, args)
                            source_paths.update(posixpath.relpath(item['path'], sb.workdir)
                                if item['path'].startswith('/') else posixpath.normpath(item['path'])
                                for item in result.get('files', []) if not item.get('error'))
                        elif name in ('list_files', 'search_files'):
                            result = self.search_repository(sb, name, args, remaining())
                        elif name == "replace_text":
                            result = self.replace_text(sb, args)
                        elif name == "write_files":
                            result = self.write_files(sb, args)
                        elif name in ("bash", "build"):
                            result = self.command(sb, args, remaining(), tool_name=name)
                            if (name == "bash" and result.get("workspace_state") and
                                    not result["workspace_state"].startswith("Working-tree status unavailable")):
                                workspace_state = result["workspace_state"]
                                if last_workspace_state is None:
                                    last_workspace_state = workspace_state
                                elif workspace_state == last_workspace_state:
                                    unchanged_bash_calls += 1
                                else:
                                    last_workspace_state = workspace_state
                                    unchanged_bash_calls = 0
                                    progress_nudge_sent = False
                                if unchanged_bash_calls >= 10 and not progress_nudge_sent:
                                    progress_checkpoint = True
                                    progress_nudge_sent = True
                            if name == "build":
                                # A failed build clears earlier success, even on identical source.
                                built_patch = None
                                if result.get("exit_code") == 0 and not result.get("timed_out"):
                                    built_patch = digest(self.patch(key))
                        elif name == "update_notes":
                            notes.append(self.append_note(directory, args["notes"], calls))
                            result = {"appended": True, "entry_count": len(notes)}
                        elif name == "submit":
                            if len(tool_calls) != 1:
                                raise ValueError("Call submit alone, after edits and build finish")
                            patch = self.patch(key)
                            if not patch.strip():
                                raise ValueError("No patch to grade; implement the fix first")
                            if digest(patch) in seen:
                                raise ValueError("This exact patch was graded already; revise it using the feedback")
                            if digest(patch) != built_patch:
                                raise ValueError("Run build on the CURRENT patch before submitting")
                            if len(grades) >= cfg.max_evaluations:
                                raise BudgetExceeded("grading allowance reached")
                            if grades:
                                self.dispatcher.continue_task(key)
                            self.dispatcher.submit_patch(key, patch)
                            event("evaluation_requested", call=calls, patch_sha256=digest(patch))
                            grade = self.dispatcher.evaluate(key)
                            last_submitted = patch
                            seen.add(digest(patch))
                            grades.append(grade)
                            (directory / f"candidate_{len(grades)}.patch").write_text(patch)
                            event("grading", **grade)
                            result = {"grading": grade,
                                "instruction":
                                "Counts only, not hidden test identities. If failing, compare the task "
                                "contract and current patch with available build/test evidence, identify "
                                "the most likely missed behavior, make a focused repair, build and submit again."}
                            passed = grade["tests_failed"] == 0
                            if not passed and len(grades) < cfg.max_evaluations:
                                latest_review = review(patch)
                                if latest_review:
                                    result['review'] = latest_review
                                    result['instruction'] += (
                                        " Independently verify the review against requirements/source. "
                                        "Apply supported repairs next; if evidence is missing, inspect only "
                                        "the named gap. Record the action or evidence-based rejection in notes.")
                        else:
                            raise ValueError(f"Unknown tool: {name}")
                    except BudgetExceeded:
                        raise
                    except DispatcherShutdown:
                        raise
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                    error = bool(result.get("error") or result.get("failures") or result.get("exit_code", 0) or result.get("timed_out")
                                 or any(item.get('error') for item in result.get('files', []) if isinstance(item, dict)))
                    if name in ('read_files', 'bash', 'build', 'list_files', 'search_files', 'replace_text', 'write_files'):
                        observation = {'call': calls, 'tool': name, 'arguments': args, 'result': result}
                        observations.append(observation)
                        try:
                            with (directory / "evidence.jsonl").open("a") as stream:
                                stream.write(json.dumps(observation, ensure_ascii=False, default=str) + "\n")
                        except OSError as exc:
                            try:
                                event("evidence_journal_error", call=calls, error=f"{type(exc).__name__}: {exc}")
                            except OSError:
                                pass
                    logger.tool_result(tool_iteration, name, result, error)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
                if progress_checkpoint:
                    messages.append({"role": "user", "content":
                        "Progress checkpoint: 10 consecutive Bash actions left the working tree "
                        "unchanged. Stop broad searching. Use the task contract and evidence already "
                        "collected to implement the smallest complete fix now. If one specific fact "
                        "blocks the edit, run one focused command, then edit. After a viable change, "
                        "inspect the diff, build the current patch, and submit it for grader feedback."})
                if passed:
                    reason = "all grader tests passed"
                    break
                if len(grades) >= cfg.max_evaluations:
                    reason = "grading allowance reached"
                    break
        except DispatcherShutdown:
            raise
        except BudgetExceeded as exc:
            reason = str(exc)
        except Exception as exc:
            reason = f"worker error: {type(exc).__name__}: {exc}"
            event("worker_error", error=reason)
            _debug(f"task {key}: {reason}")
        # extract_patch can overwrite the dispatcher's patch file during tools.
        # Never let done() grade unsubmitted work: retain only the last graded patch.
        self.dispatcher.submit_patch(key, last_submitted)
        event("finished", reason=reason, calls=calls, grades=grades, budget=self.budget.snapshot(key))
        return self.dispatcher.done(key, reason=reason, iterations=calls)
