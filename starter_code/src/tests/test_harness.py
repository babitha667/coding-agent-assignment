"""Offline harness tests. No course API, Docker, credentials or hidden tests."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

STARTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STARTER))
from src.agentic_loop import Agent, SOLVER_PROMPT, VERIFICATION_PROMPTS, compact_notebook, final_review_report
from src.config import CONFIG
from src.cost_budget import BudgetExceeded, CostBudget
from src.task_state import TaskState, digest
from src.tools import TaskTools, static_checks


import test_workflow as support
from src.agentic_loop import user as user_message
from test_workflow import (LocalSandbox, NullLogger, completion, action, repair_review,
                           record_repair, review_patch, ScriptedClient, FakeDispatcher)


class HarnessTests(unittest.TestCase):
    # Legacy reference-mode fixtures exercise unchanged accounting, source tools,
    # static readiness, restoration and concurrency. Production defaults are tested separately.
    def setUp(self):
        support.WorkflowTests.setUp(self)
        self.cfg = replace(self.cfg, verification_mode='reference', review_enabled=False,
                           expert_review_calls=1, review_format_retries=0, expert_calls=3,
                           expert_budget_usd=.18, max_model_calls=20)
        self.state = TaskState(self.task, self.cfg)
        self.state.role_calls = {}
        self.tools = TaskTools(LocalSandbox('offline', self.repo), self.state, self.cfg)

    agent = support.WorkflowTests.agent
    edit = support.WorkflowTests.edit
    smoke_arguments = staticmethod(support.WorkflowTests.smoke_arguments)
    ready = support.WorkflowTests.ready

    def failed_fixture(self):
        agent = self.agent()
        self.state.update_notes({'plan': ['calc.py:triple: fix the positive example'],
                                 'scenarios': ['0 -> 0; preserve zero behavior']})
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 2\n')
        patch = agent.current_patch(self.state)
        self.ready(agent)
        feedback = agent.grade(self.state, self.tools, patch)
        self.assertEqual(feedback['grading']['tests_failed'], 1)
        return agent, feedback

    def test_two_attempts_keep_context_and_pass(self):
        agent = self.agent({'standard': [action('read_files', requests=[{'path': 'calc.py'}]),
            action('update_notes', findings=['calc.py:2 multiplies by 1'], plan=['Multiply by 3'],
                   scenarios=['0 -> 0; -2 -> -6; 2 -> 6']), self.edit(2), action('submit'),
            record_repair(), self.edit(3), review_patch(), action('submit')]})
        result = agent.solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        self.assertEqual(len(self.dispatcher.results), 2)
        self.assertEqual(self.dispatcher.continues, 1)
        repair_context = json.dumps(self.client.requests[-1]['messages'])
        self.assertIn('calc.py:2 multiplies by 1', repair_context)
        self.assertIn('Counts only', repair_context)
        notebook = json.loads((self.home / 'logs/0/working_state.json').read_text())
        self.assertEqual(len(notebook['grades']), 2)
        self.assertNotIn('working_state.json', self.dispatcher.submissions[-1])

    def test_regression_restores_best_with_counted_grade(self):
        self.dispatcher.grade_script = [1, 4, 1]
        self.agent({'standard': [self.edit(2), action('submit'), record_repair(),
                    self.edit(4), review_patch(), action('submit')]}).solve(self.task)
        self.assertEqual(len(self.dispatcher.results), 3)
        self.assertEqual(self.dispatcher.final['tests_failed'], 1)
        self.assertIn('+    return x * 2', self.dispatcher.submissions[-1])
        self.assertEqual(self.dispatcher.continues, 2)
        baseline = json.loads((self.home / 'logs/0/candidate_1.review.json').read_text())
        restored = json.loads((self.home / 'logs/0/candidate_3.review.json').read_text())
        self.assertTrue(restored['restoration'])
        self.assertEqual(restored['notes'], baseline['notes'])

    def test_no_empty_patch_automatic_grade(self):
        cfg = replace(self.cfg, max_model_calls=2)
        self.agent({'standard': [completion('I should inspect'), completion('I should inspect')]}, cfg).solve(self.task)
        self.assertEqual(len(self.dispatcher.results), 0)
        self.assertEqual(self.dispatcher.patchfile.read_text(), '')

    def test_orientation_bounded_and_read_only(self):
        cfg = replace(self.cfg, orientation_calls=3)
        self.agent({'starter': [action('bash', command='touch forbidden'),
                    action('read_files', requests=[{'path': 'README.md'}]),
                    action('update_notes', repo_map='calc.py contains triple')],
                    'standard': [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertFalse((self.repo / 'forbidden').exists())
        self.assertEqual(sum(r['model'] == 'starter' for r in self.client.requests), 3)
        self.assertIn('calc.py contains triple', json.dumps(self.client.requests[3]['messages']))

    def test_default_standard_configuration_implements_without_starter_orientation(self):
        cfg = replace(CONFIG, log_dir=str(self.home / 'logs'), review_enabled=False, workers=1)
        regression = action('bash', command="python3 -B -c 'from calc import triple; assert triple(-2)==-6'",
                            purpose='test', covers=['calc.py'])
        self.agent({'standard': [self.edit(3), regression, action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        self.assertTrue(all(r['model'] == 'standard' for r in self.client.requests))

    def test_truncated_tool_call_never_executes(self):
        cfg = replace(self.cfg, max_model_calls=2)
        bad = completion(calls=[('bash', {'command': 'touch forbidden'})], finish='length')
        agent = self.agent({'standard': [bad, completion('No safe action')]}, cfg)
        agent.solve(self.task)
        self.assertFalse((self.repo / 'forbidden').exists())
        self.assertIn('no partial tools were executed', json.dumps(self.client.requests[-1]['messages']))

    def test_failed_request_is_charged_and_sdk_retry_counted(self):
        agent = self.agent({'standard': [TimeoutError('offline timeout'), completion('ok')]})
        response = agent.request(self.state, 'standard', 'solver', [{'role': 'user', 'content': 'Hi'}], [], 1000)
        self.assertEqual(response['content'], 'ok')
        snapshot = agent.budget.snapshot(0)
        self.assertEqual((snapshot['calls'], snapshot['unknown_requests']), (2, 1))
        self.assertGreater(snapshot['spent'], 0)
        self.assertEqual(snapshot['reserved'], 0)

    def test_role_limit_includes_retry(self):
        agent = self.agent({'expert': [TimeoutError('offline timeout')]})
        with self.assertRaises(BudgetExceeded):
            agent.request(self.state, 'expert', 'review', [user_message('Hi')], [], 1000, role_limit=1)
        self.assertEqual(self.state.calls, 1)

    def test_batch_reads_clamp_and_return_continuation(self):
        (self.repo / 'long.txt').write_text('line\n' * 900)
        data = self.tools.read_files({'requests': [{'path': 'long.txt', 'num_lines': 50000}]})['files'][0]
        self.assertEqual(data['next_start_line'], 501)
        self.assertEqual(data['total_lines'], 900)
        data = self.tools.read_files({'path': 'long.txt', 'start_line': 501, 'num_lines': 900})['files'][0]
        self.assertIsNone(data['next_start_line'])

    def test_long_single_line_is_explicit_not_silently_complete(self):
        (self.repo / 'long.txt').write_text('x' * 50000)
        data = self.tools.read_files({'path': 'long.txt'})['files'][0]
        self.assertEqual(data['next_start_line'], 1)
        self.assertIn('LINE TRUNCATED', data['content'])

    def test_read_batch_more_than_four_files_and_pending(self):
        requests = [{'path': 'calc.py'} for _ in range(25)]
        data = self.tools.read_files({'requests': requests})
        self.assertGreater(len(data['files']), 4)
        self.assertEqual(len(data['files']) + len(data['pending_requests']), 25)

    def test_source_paths_and_symlinks_cannot_escape(self):
        outside = self.home / 'outside.txt'
        outside.write_text('not a repository source')
        (self.repo / 'escape').symlink_to(outside)
        for path in ('../outside.txt', 'escape'):
            result = self.tools.read_files({'path': path})['files'][0]
            self.assertIn('error', result)
            self.assertNotIn('content', result)

    def test_evidence_stale_after_edit_and_refreshed_by_read(self):
        self.tools.read_files({'path': 'calc.py'})
        self.state.observe_patch('changed source')
        self.assertTrue(self.state.notebook()['source_evidence']['calc.py']['needs_refresh'])
        self.tools.read_files({'path': 'calc.py'})
        self.assertFalse(self.state.notebook()['source_evidence']['calc.py']['needs_refresh'])

    def test_json_string_notes_and_bounded_evidence(self):
        self.state.update_notes({'plan': '["inspect call site", "fix multiplier"]'})
        self.assertEqual(len(self.state.notes['plan']), 2)
        for n in range(40):
            self.state.record_read(f'{n}.py', str(n), 1, 2, 'excerpt')
        self.assertEqual(len(self.state.evidence), 32)
        self.assertLess(len(compact_notebook(self.state)), 26001)

    def test_reference_test_guard_and_valid_multiline_edit(self):
        result = self.tools.bash({'command': 'pytest -q'})
        self.assertIn('error', result)
        command = "python3 - <<'PY'\nfrom pathlib import Path\nPath('new.py').write_text('x = 1\\n')\nPY"
        result = self.tools.bash({'command': command, 'purpose': 'edit'})
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue((self.repo / 'new.py').exists())

    def test_captured_multiline_edits_are_valid_shell(self):
        payload = Path(__file__).with_name('handoff_payloads_20261016.json')
        if not payload.exists():
            self.skipTest('Historical payload fixtures unavailable')
        commands = [p['arguments']['command'] for p in json.loads(payload.read_text())
                    if p['arguments'].get('command')]
        self.assertGreaterEqual(len(commands), 5)
        for command in commands:
            result = subprocess.run(['bash', '-n'], input=command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_static_failure_not_submitted(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def broken(:\n')
        patch = agent.current_patch(self.state)
        self.assertFalse(static_checks(self.tools.sb)['ok'])
        self.assertIn('error', agent.grade(self.state, self.tools, patch))
        self.assertFalse(self.dispatcher.results)

    def test_missing_import_in_changed_helper_blocks_grading(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return re.escape(str(x))\n')
        patch = agent.current_patch(self.state)
        checks = static_checks(self.tools.sb)
        self.assertFalse(checks['ok'])
        self.assertEqual(checks['undefined_globals'][0]['name'], 're')
        self.assertIn('static-check', agent.grade(self.state, self.tools, patch)['error'])
        self.assertEqual(self.state.evaluation_attempts, 0)
        self.assertFalse(self.dispatcher.results)
        (self.repo / 'calc.py').write_text('import re\ndef triple(x):\n    return re.escape(str(x))\n')
        self.assertTrue(static_checks(self.tools.sb)['ok'])

    def test_compiler_rejects_errors_that_ast_parse_accepts(self):
        (self.repo / 'calc.py').write_text('return 3\n')
        self.agent().current_patch(self.state)
        checks = static_checks(self.tools.sb)
        self.assertFalse(checks['ok'])
        self.assertIn('outside function', checks['python_errors'][0])

    def test_compiler_warnings_do_not_corrupt_validation_response(self):
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x is 3\n')
        self.agent().current_patch(self.state)
        checks = static_checks(self.tools.sb)
        self.assertTrue(checks['ok'], checks)
        self.assertTrue(checks['warnings'])

    def test_name_check_understands_imports_builtins_closures_and_class_scope(self):
        (self.repo / 'calc.py').write_text('import re as rx\n'
            'def triple(x):\n    def inner():\n        return x * 3\n    return inner()\n'
            'class Checker:\n    def check(self, value):\n        return rx.escape(str(len(value)))\n')
        self.agent().current_patch(self.state)
        checks = static_checks(self.tools.sb)
        self.assertTrue(checks['ok'], checks)

    def test_unchanged_baseline_name_issue_does_not_block_unrelated_fix(self):
        (self.repo / 'calc.py').write_text('def legacy():\n    return injected_by_runtime\n'
                                         'def triple(x):\n    return x * 1\n')
        subprocess.run(['git', 'add', 'calc.py'], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'baseline dynamic name'], cwd=self.repo, check=True, capture_output=True)
        source = (self.repo / 'calc.py').read_text().replace('x * 1', 'x * 3')
        (self.repo / 'calc.py').write_text(source)
        self.agent().current_patch(self.state)
        self.assertTrue(static_checks(self.tools.sb)['ok'])

    def test_review_text_and_syntax_success_do_not_replace_smoke_execution(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.state.update_notes({'review': 'All scenarios PASS; code is runnable.'})
        command = completion(calls=[('bash', {'command': 'git diff --check', 'purpose': 'check'})])
        agent.execute_tools(self.state, self.tools, command['choices'][0]['message'], {'bash'})
        feedback = agent.grade(self.state, self.tools, patch)
        self.assertEqual(feedback['missing_checks']['smoke'], ['calc.py'])
        self.assertFalse(self.dispatcher.results)
        self.ready(agent)
        self.assertIn('grading', agent.grade(self.state, self.tools, patch))

    def test_smoke_executes_changed_function_and_reports_runtime_failure(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return (1).missing_attribute\n')
        patch = agent.current_patch(self.state)
        self.assertTrue(static_checks(self.tools.sb)['ok'])
        self.ready(agent)
        feedback = agent.grade(self.state, self.tools, patch)
        self.assertIn('AttributeError', feedback['recent_checks'][-1]['output'])
        self.assertFalse(feedback['recent_checks'][-1]['passed'])
        self.assertEqual(self.state.evaluation_attempts, 0)

    def test_edit_invalidates_successful_readiness_check(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        agent.current_patch(self.state)
        self.ready(agent)
        self.assertTrue(self.state.readiness_checks[-1]['passed'])
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 4\n')
        patch = agent.current_patch(self.state)
        self.assertFalse(self.state.readiness_checks)
        self.assertIn('missing_checks', agent.grade(self.state, self.tools, patch))

    def test_latest_failed_smoke_check_overrides_previous_pass(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.ready(agent)
        bad = self.smoke_arguments("python3 -B -c 'raise RuntimeError(\"smoke failed\")'")
        message = completion(calls=[('bash', bad)])['choices'][0]['message']
        agent.execute_tools(self.state, self.tools, message, {'bash'})
        self.assertIn('missing_checks', agent.grade(self.state, self.tools, patch))

    def test_compiled_source_requires_build_as_well_as_smoke_check(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('No host C compiler for this offline fixture')
        import shlex
        agent = self.agent()
        (self.repo / 'demo.c').write_text('int main(void) { return 0; }\n')
        patch = agent.current_patch(self.state)
        binary = self.home / 'demo'
        compile_command = f'{shlex.quote(compiler)} demo.c -o {shlex.quote(str(binary))}'
        smoke = self.smoke_arguments(compile_command + ' && ' + shlex.quote(str(binary)), ['demo.c'])
        agent.execute_tools(self.state, self.tools,
                            completion(calls=[('bash', smoke)])['choices'][0]['message'], {'bash'})
        feedback = agent.grade(self.state, self.tools, patch)
        self.assertEqual(feedback['missing_checks']['smoke'], [])
        self.assertEqual(feedback['missing_checks']['build'], ['demo.c'])
        build = self.smoke_arguments(compile_command, ['demo.c'], 'build')
        agent.execute_tools(self.state, self.tools,
                            completion(calls=[('bash', build)])['choices'][0]['message'], {'bash'})
        self.assertIn('grading', agent.grade(self.state, self.tools, patch))

    def test_readiness_metadata_rejects_paths_outside_checkout(self):
        with self.assertRaises(ValueError):
            self.tools.bash(self.smoke_arguments(covers=['../outside.py']))

    def test_trivial_or_syntax_only_commands_do_not_count_as_readiness(self):
        for command in ('true', 'echo PASS', 'git diff --check', 'python3 -m py_compile calc.py'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.tools.bash(self.smoke_arguments(command))

    def test_readiness_gate_can_be_disabled_but_static_errors_still_block(self):
        cfg = replace(self.cfg, require_code_readiness=False)
        agent = self.agent(cfg=cfg)
        (self.repo / 'calc.py').write_text('def triple(x):\n    return missing_name\n')
        patch = agent.current_patch(self.state)
        self.assertIn('static-check', agent.grade(self.state, self.tools, patch)['error'])
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.assertIn('grading', agent.grade(self.state, self.tools, patch))

    def test_duplicate_patch_not_regraded(self):
        agent = self.agent()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.ready(agent)
        agent.grade(self.state, self.tools, patch)
        self.assertIn('graded already', agent.grade(self.state, self.tools, patch)['error'])
        self.assertEqual(len(self.dispatcher.results), 1)

    def test_optional_local_test_gate_invalidated_by_edit(self):
        cfg = replace(self.cfg, verification_mode='local_tests')
        agent = self.agent(cfg=cfg)
        state = TaskState(self.task, cfg)
        tools = TaskTools(self.tools.sb, state, cfg)
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(state)
        state.observe_patch(patch)
        self.ready(agent, state, tools)
        self.assertIn('local tests', agent.grade(state, tools, patch)['error'])
        message = {'tool_calls': [{'id': 't', 'function': {'name': 'bash', 'arguments': json.dumps({
            'command': 'python3 -c "from calc import triple; assert triple(-2) == -6"',
            'purpose': 'test', 'covers': ['calc.py']})}}]}
        # Avoid a fixture bytecode file being mistaken for a source edit.
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        self.addCleanup(os.environ.pop, 'PYTHONDONTWRITEBYTECODE', None)
        agent.execute_tools(state, tools, message, {'bash'})
        self.assertTrue(state.last_local_test['passed'])
        self.assertIn('grading', agent.grade(state, tools, patch))
        state.observe_patch(patch + 'new edit')
        self.assertIsNone(state.last_local_test)

    def test_compaction_never_orphans_tool_results(self):
        cfg = replace(self.cfg, context_max_chars=8000, context_recent_exchanges=2)
        agent = self.agent(cfg=cfg)
        messages = [{'role': 'system', 'content': 'instructions'}, user_message('issue')]
        for n in range(12):
            messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(n)}]},
                             {'role': 'tool', 'tool_call_id': str(n), 'content': 'x' * 2000}])
        reduced = agent.compact(self.state, messages)
        self.assertLess(len(json.dumps(reduced)), 8000)
        self.assertEqual(reduced[3]['role'], 'assistant')

    def test_compaction_handles_fewer_exchanges_than_retention_limit(self):
        # Live call 107 exceeded the context bound with only five exchanges;
        # selecting starts[-6] used to terminate the entire worker.
        cfg = replace(self.cfg, context_max_chars=8000, context_recent_exchanges=6)
        agent = self.agent(cfg=cfg)
        for count in range(1, 6):
            with self.subTest(exchanges=count):
                messages = [{'role': 'system', 'content': 'instructions'}, user_message('issue')]
                for n in range(count):
                    messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(n)}]},
                                     {'role': 'tool', 'tool_call_id': str(n), 'content': 'x' * 10000}])
                reduced = agent.compact(self.state, messages)
                self.assertLess(len(json.dumps(reduced)), 8000)
                active = set()
                for message in reduced:
                    if message['role'] == 'assistant':
                        active = {call['id'] for call in message.get('tool_calls', [])}
                    elif message['role'] == 'tool':
                        self.assertIn(message['tool_call_id'], active)

    def test_worker_errors_include_traceback_for_future_diagnosis(self):
        def callback(request):
            raise RuntimeError('offline diagnostic fixture')
        self.agent(callback=callback).solve(self.task)
        events = [json.loads(row) for row in (self.home / 'logs/0/events.jsonl').read_text().splitlines()]
        error = next(row for row in events if row['event'] == 'worker_error')
        self.assertIn('Traceback', error['traceback'])
        self.assertIn('offline diagnostic fixture', error['traceback'])

    def test_concurrent_calls_share_api_slots_and_budget(self):
        active = peak = 0
        lock = threading.Lock()
        def callback(request):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.02)
            with lock:
                active -= 1
            return completion('ok')
        cfg = replace(self.cfg, api_concurrency=2)
        agent = self.agent(cfg=cfg, callback=callback)
        states = [TaskState({**self.task, 'task': k}, cfg) for k in range(5)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(agent.request, st, 'standard', 'solver', [user_message('Hi')], [], 1000)
                       for st in states]
            for future in futures:
                self.assertEqual(future.result()['content'], 'ok')
        self.assertEqual(peak, 2)
        self.assertEqual(agent.budget.reserved, 0)
        self.assertEqual(sum(agent.budget.snapshot(s.key)['calls'] for s in states), 5)

    def test_five_workers_complete_isolated_tasks(self):
        dispatchers, tasks = {}, []
        for key in range(5):
            repo = self.home / f'repo-{key}'
            shutil.copytree(self.repo, repo)
            dispatchers[key] = FakeDispatcher(repo)
            tasks.append({**self.task, 'task': key, 'workdir': str(repo)})
        class MultiDispatcher:
            def __init__(self):
                self.pending = list(tasks)
                self.lock = threading.Lock()
            def next_task(self):
                with self.lock:
                    return self.pending.pop(0) if self.pending else None
            def __getattr__(self, name):
                return lambda key, *args, **kwargs: getattr(dispatchers[key], name)(key, *args, **kwargs)
        steps, lock = {}, threading.Lock()
        def callback(request):
            task = json.loads(request['messages'][1]['content'].split('\nInitial notebook:')[0])
            with lock:
                step = steps.get(task['workdir'], 0)
                steps[task['workdir']] = step + 1
            return self.edit(3) if step == 0 else action('submit')
        cfg = replace(self.cfg, workers=5, api_concurrency=3)
        client = ScriptedClient(callback=callback)
        agent = Agent(MultiDispatcher(), cfg, client, LocalSandbox)
        agent.run()
        self.assertEqual(len(client.requests), 10)
        self.assertTrue(all(d.final['tests_failed'] == 0 for d in dispatchers.values()))
        self.assertTrue(all('+    return x * 3' in d.submissions[-1] for d in dispatchers.values()))

    def test_repeated_reads_recover_instead_of_fatal_discovery_gate(self):
        reads = [action('read_files', requests=[{'path': 'calc.py'}])] * 8
        result = self.agent({'standard': reads + [self.edit(3), action('submit')]}).solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        self.assertIn('No new source evidence', json.dumps(self.client.requests[-1]['messages']))

    def test_unsubmitted_patch_is_not_graded_at_call_limit(self):
        cfg = replace(self.cfg, max_model_calls=1)
        self.agent({'standard': [self.edit(3)]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 8)
        self.assertEqual(len(self.dispatcher.results), 0)

    def test_time_limit_prevents_api_admission(self):
        agent = self.agent()
        self.state.started -= self.cfg.task_timeout_s + 1
        with self.assertRaises(BudgetExceeded):
            agent.request(self.state, 'standard', 'solver', [user_message('Hi')], [], 1000)
        self.assertEqual(len(self.client.requests), 0)

    def test_expert_cap_is_shared_across_roles(self):
        agent = self.agent({'expert': [completion('ok')] * 3})
        for role in ('solver', 'review', 'orientation'):
            agent.request(self.state, 'expert', role, [user_message('Hi')], [], 1000)
        with self.assertRaises(BudgetExceeded):
            agent.request(self.state, 'expert', 'solver', [user_message('Hi')], [], 1000)
        self.assertEqual(agent.budget.snapshot(0)['expert_calls'], 3)

    def test_solver_can_be_changed_to_starter_only(self):
        cfg = replace(self.cfg, solver_model='starter')
        self.agent({'starter': [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        self.assertTrue(all(r['model'] == 'starter' for r in self.client.requests))

    def test_grading_error_still_reserves_counted_restoration(self):
        agent = self.agent()
        self.dispatcher.grade_script = [1, 1]
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 2\n')
        patch = agent.current_patch(self.state)
        self.ready(agent)
        agent.grade(self.state, self.tools, patch)
        original = self.dispatcher.evaluate
        def fail_once(key):
            self.dispatcher.evaluate = original
            raise RuntimeError('offline grading transport failure')
        self.dispatcher.evaluate = fail_once
        self.state.update_notes({'repair_review': repair_review()})
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 4\n')
        patch = agent.current_patch(self.state)
        self.state.update_notes({'review': 'Assessed the changed code against old and new cases.'})
        self.ready(agent)
        with self.assertRaises(RuntimeError):
            agent.grade(self.state, self.tools, patch)
        agent.finish(self.state, self.tools, 'offline transport fixture')
        self.assertEqual(self.state.evaluation_attempts, 3)
        self.assertEqual(self.dispatcher.final['tests_failed'], 1)
        self.assertIn('+    return x * 2', self.dispatcher.submissions[-1])

    def test_no_restore_slot_does_not_misreport_best_as_final(self):
        cfg = replace(self.cfg, max_evaluations=2, reserve_restoration_grade=False)
        self.dispatcher.grade_script = [1, 4]
        self.agent({'standard': [self.edit(2), action('submit'), record_repair(),
                    self.edit(4), review_patch(), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 4)
        self.assertEqual(len(self.dispatcher.results), 2)
        self.assertIn('+    return x * 2', (self.home / 'logs/0/best.patch').read_text())

    def test_failed_grade_snapshots_previous_plan_and_cases(self):
        _, feedback = self.failed_fixture()
        self.state.update_notes({'plan': ['new plan'], 'scenarios': ['-2 -> -6']})
        self.assertEqual(feedback['previous_cycle']['plan'], ['calc.py:triple: fix the positive example'])
        snapshot = json.loads((self.home / 'logs/0/candidate_1.review.json').read_text())
        self.assertEqual(snapshot['notes']['scenarios'], ['0 -> 0; preserve zero behavior'])
        self.assertEqual(snapshot['verification_mode'], 'reference')
        self.assertTrue(self.state.needs_repair_review)

    def test_regrading_requires_new_case_and_review_of_current_patch(self):
        agent, _ = self.failed_fixture()
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.assertIn('repair_review', agent.grade(self.state, self.tools, patch)['error'])
        self.assertEqual(len(self.dispatcher.results), 1)
        self.state.update_notes({'repair_review': repair_review()})
        self.assertIn('CURRENT code', agent.grade(self.state, self.tools, patch)['error'])
        self.state.update_notes({'review': 'Current x * 3 yields -6 for -2 and 0 for 0; signature is unchanged.'})
        self.ready(agent)
        self.assertEqual(agent.grade(self.state, self.tools, patch)['grading']['tests_failed'], 0)

    def test_edit_invalidates_previous_scenario_review(self):
        self.failed_fixture()
        self.state.update_notes({'repair_review': repair_review(), 'review': 'Reviewed old patch; it has a gap.'})
        self.assertIsNotNone(self.state.scenario_review_patch)
        self.state.observe_patch('new source revision')
        self.assertIsNone(self.state.scenario_review_patch)

    def test_repair_keeps_prior_cases_and_records_why_old_patch_fails(self):
        self.failed_fixture()
        self.state.update_notes({'repair_review': repair_review()})
        self.assertIn('0 -> 0; preserve zero behavior', self.state.notes['scenarios'])
        review = self.state.repair_review
        self.assertEqual(review['for_patch_sha256'], self.state.latest.fingerprint)
        self.assertIn('-4', review['new_scenarios'][0]['previous_patch_behavior'])
        self.assertEqual(self.state.notes['plan'], review['planned_changes'])

    def test_repair_rejects_empty_or_reused_counterexamples(self):
        self.failed_fixture()
        invalid = repair_review()
        invalid['new_scenarios'] = []
        with self.assertRaisesRegex(ValueError, 'counterexamples'):
            self.state.update_notes({'repair_review': invalid})
        self.state.update_notes({'repair_review': repair_review()})
        self.state.record_grade('another graded source patch', {'tests_failed': 1, 'tests_total': 8, 'attempt': 1})
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            self.state.update_notes({'repair_review': repair_review()})
        self.state.update_notes({'repair_review': repair_review('-3', '-9')})

    def test_current_cycle_review_can_be_refined(self):
        self.failed_fixture()
        self.state.update_notes({'repair_review': repair_review()})
        refined = repair_review()
        refined['planned_changes'].append('calc.py:triple: keep the function signature unchanged.')
        self.state.update_notes({'repair_review': refined})
        self.assertEqual(len(self.state.notes['plan']), 2)

    def test_repair_checkpoint_keeps_editing_and_checks_available(self):
        self.agent({'standard': [self.edit(2), action('submit'), record_repair(),
                    self.edit(3), review_patch(), action('submit')]}).solve(self.task)
        checkpoint = self.client.requests[2]
        self.assertEqual(checkpoint['model'], 'standard')
        self.assertEqual(checkpoint['tool_choice'], 'auto')
        self.assertEqual({t['function']['name'] for t in checkpoint['tools']},
                         {'bash', 'read_files', 'run_tests', 'update_notes', 'submit'})
        self.assertEqual(checkpoint['max_tokens'], self.cfg.output_limit('standard'))
        self.assertIn('previous_patch_behavior', json.dumps(checkpoint))
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_counterexample_edit_smoke_and_assessment_can_share_first_repair_response(self):
        edit_calls = [(call['function']['name'], json.loads(call['function']['arguments']))
                      for call in self.edit(3)['choices'][0]['message']['tool_calls']]
        repair = completion(calls=[('update_notes', {'repair_review': repair_review()})] + edit_calls +
            [('update_notes', {'review': 'Traced the changed multiplier through old and new scenarios.'})])
        self.agent({'standard': [self.edit(2), action('submit'), repair, action('submit')]}).solve(self.task)
        self.assertEqual(len(self.client.requests), 4)
        self.assertEqual([r['tests_failed'] for r in self.dispatcher.results], [1, 0])
        self.assertIn('FAILED-GRADING REPAIR', json.dumps(self.client.requests[2]['messages']))
        saved = json.loads((self.home / 'logs/0/candidate_2.review.json').read_text())
        self.assertEqual(saved['repair_review']['previous_assumption'], repair_review()['previous_assumption'])

    def test_repair_requirement_is_configurable(self):
        cfg = replace(self.cfg, require_repair_review=False)
        self.agent({'standard': [self.edit(2), action('submit'), self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_no_edit_recovery_can_lead_to_a_patch(self):
        cfg = replace(self.cfg, inspection_turn_limit=3, inspection_recovery_turns=2)
        step = {'action': 'implement', 'target': 'calc.py:triple',
                'reason': 'The function multiplies by 1 but the task requires 3.',
                'change_or_question': 'Replace x * 1 with x * 3; preserve the signature.'}
        self.agent({'standard': [action('read_files', path='calc.py')] * 3 +
                    [action('update_notes', next_step=step), self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        self.assertEqual(self.client.requests[3]['tool_choice'], 'auto')
        self.assertEqual({tool['function']['name'] for tool in self.client.requests[3]['tools']},
                         {'bash', 'read_files', 'run_tests', 'update_notes', 'submit'})
        self.assertIn('NO-EDIT CHECKPOINT', json.dumps(self.client.requests[3]))

    def test_new_notes_do_not_delay_early_implementation_reminder_or_remove_tools(self):
        script = [action('update_notes', findings=[f'Fresh observation {i}'],
                         plan=['calc.py:triple replace x * 1 with x * 3']) for i in range(6)]
        self.agent({'standard': script + [self.edit(3), action('submit')]}).solve(self.task)
        next_request = self.client.requests[6]
        self.assertIn('IMPLEMENTATION CHECKPOINT', json.dumps(next_request['messages']))
        self.assertIn('calc.py:triple replace x * 1 with x * 3', json.dumps(next_request['messages']))
        self.assertEqual(next_request['tool_choice'], 'auto')
        self.assertEqual({t['function']['name'] for t in next_request['tools']},
                         {'bash', 'read_files', 'run_tests', 'update_notes', 'submit'})
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_solver_reuses_orientation_source_excerpts(self):
        cfg = replace(self.cfg, orientation_calls=1)
        self.agent({'starter': [action('read_files', requests=[{'path': 'calc.py'}])],
                    'standard': [self.edit(3), action('submit')]}, cfg).solve(self.task)
        initial = self.client.requests[1]['messages'][1]['content']
        self.assertIn('Source already read during orientation', initial)
        self.assertIn('return x * 1', initial)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_review_packet_bounds_and_marks_large_sections_without_dropping_categories(self):
        agent = self.agent()
        self.state.notes['plan'] = ['P' * 50000]
        self.state.notes['scenarios'] = ['T' * 50000]
        packet = agent.review_packet(self.state, {'content': 'S' * 50000}, 'D' * 50000, 'failed grading')
        self.assertLessEqual(len(packet), self.cfg.review_context_chars)
        for label in ('Implementation plan:', 'Verification cases and assessment:',
                      'Executed checks (not predicted results):', 'Inspection findings', 'Current source:'):
            self.assertIn(label, packet)
        self.assertIn('[EXCERPT INCOMPLETE;', packet)

    def test_grading_ledger_preserves_each_round_diagnosis_cases_checks_and_patch(self):
        self.state.update_notes({'hypotheses': ['Round zero: negative values use wrong multiplier'],
                                 'plan': ['calc.py:triple correct multiplier'], 'scenarios': ['-2 -> -6']})
        self.state.readiness_checks = [{'command': 'smoke_case', 'output': 'observed result', 'passed': True}]
        self.state.record_grade('patch zero', {'tests_failed': 4, 'tests_total': 8, 'attempt': 0})
        self.state.update_notes({'hypotheses': ['Round one: zero boundary remains uncertain']})
        self.state.record_grade('patch one', {'tests_failed': 4, 'tests_total': 8, 'attempt': 1})
        first = self.state.read_round(0)
        self.assertEqual(first['issue_reasoned']['hypotheses'], ['Round zero: negative values use wrong multiplier'])
        self.assertEqual(first['notes']['scenarios'], ['-2 -> -6'])
        self.assertEqual(self.state.read_round(0, 'checks')['checks'][0]['output'], 'observed result')
        self.assertEqual(self.state.read_round(0, 'patch')['patch'], 'patch zero')
        self.assertEqual([r['status'] for r in self.state.round_history], ['unresolved', 'unresolved'])
        self.assertEqual(len((self.state.directory / 'grading_ledger.jsonl').read_text().splitlines()), 2)
        for invalid in (-1, '0', '../../secrets', True):
            with self.assertRaises(ValueError):
                self.state.read_round(invalid)

    def test_all_rounds_remain_in_review_packet_with_paginated_detail_retrieval(self):
        for attempt in range(9):
            self.state.update_notes({'hypotheses': [f'Issue in round {attempt}: ' + 'X' * 1000],
                                     'plan': ['Detailed change ' + 'Y' * 1000] * 24})
            self.state.record_grade(f'patch {attempt}', {'tests_failed': 3, 'tests_total': 8, 'attempt': attempt})
        packet = self.agent().review_packet(self.state, {}, '', 'failed grading')
        self.assertLessEqual(len(packet), self.cfg.review_context_chars)
        for attempt in range(9):
            self.assertIn(f'read_round(attempt={attempt})', packet)
        chunks, offset = [], 0
        while True:
            result = self.tools.invoke('read_round', {'attempt': 0, 'section': 'diagnosis', 'offset': offset})
            chunks.append(result['content'])
            if result['next_offset'] is None:
                break
            offset = result['next_offset']
        self.assertEqual(json.loads(''.join(chunks)), self.state.read_round(0))

    def test_misformatted_recovery_notes_cannot_lock_out_edits(self):
        cfg = replace(self.cfg, inspection_turn_limit=3, inspection_recovery_turns=3)
        # Reproduce the live run: next_step fields appeared at the top level.
        flat_step = action('update_notes', action='implement', target='calc.py:triple',
                           reason='The multiplier is 1 instead of 3.',
                           change_or_question='Replace x * 1 with x * 3.')
        self.agent({'standard': [action('read_files', path='calc.py')] * 3 +
                    [flat_step, self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        for request in self.client.requests[3:]:
            self.assertEqual(request['tool_choice'], 'auto')
            self.assertIn('bash', [tool['function']['name'] for tool in request['tools']])

    def test_default_windows_allow_75_no_edit_solver_turns_not_25_total_calls(self):
        cfg = replace(self.cfg, max_model_calls=100, stop_on_no_edit=True)
        def callback(request):
            return action('update_notes', findings=[f'Observation {len(self.client.requests)}'])
        self.agent(cfg=cfg, callback=callback).solve(self.task)
        self.assertEqual(len(self.client.requests), 75)
        self.assertTrue(all(request['tool_choice'] == 'auto' for request in self.client.requests))
        self.assertFalse(self.dispatcher.results)
        events = [json.loads(row) for row in (self.home / 'logs/0/events.jsonl').read_text().splitlines()]
        checkpoints = [row for row in events if row['event'] == 'inspection_checkpoint']
        self.assertEqual([row['solver_turns'] for row in checkpoints], [25, 25])

    def test_endless_rereading_stops_without_forcing_a_guessed_patch(self):
        cfg = replace(self.cfg, max_model_calls=40, inspection_turn_limit=3, inspection_recovery_turns=2,
                      stop_on_no_edit=True)
        def callback(request):
            if request['tool_choice'] != 'auto':
                return action('update_notes', next_step={'action': 'inspect', 'target': 'calc.py:triple',
                    'reason': 'Need to confirm the current multiplier.',
                    'change_or_question': 'What multiplier does triple currently use?'})
            return action('read_files', path='calc.py')
        self.agent(cfg=cfg, callback=callback).solve(self.task)
        self.assertEqual(len(self.client.requests), 7)
        self.assertFalse(self.dispatcher.results)
        events = [json.loads(row) for row in (self.home / 'logs/0/events.jsonl').read_text().splitlines()]
        self.assertIn('inspection_loop', events[-1]['reason'].replace(' ', '_'))
        self.assertEqual(sum(e['event'] == 'inspection_checkpoint' for e in events), 2)

    def test_cosmetic_notes_do_not_reset_no_edit_counter(self):
        cfg = replace(self.cfg, max_model_calls=40, inspection_turn_limit=3, inspection_recovery_turns=2,
                      stop_on_no_edit=True)
        def callback(request):
            if request['tool_choice'] != 'auto':
                return action('update_notes', next_step={'action': 'implement', 'target': 'calc.py:triple',
                    'reason': 'Multiplier is wrong.', 'change_or_question': 'Change the multiplier to 3.'})
            return action('update_notes', findings=[f'Another observation {len(self.client.requests)}'])
        self.agent(cfg=cfg, callback=callback).solve(self.task)
        self.assertEqual(len(self.client.requests), 7)
        self.assertFalse(self.dispatcher.results)

    def test_unchanged_edit_command_is_not_patch_progress(self):
        self.state.no_edit_turns = 3
        self.state.track_edit_progress(self.state.patch_hash)
        self.assertEqual(self.state.no_edit_turns, 4)
        self.state.observe_patch('actual diff')
        self.state.inspection_recoveries = 2
        self.state.track_edit_progress(digest(''))
        self.assertEqual((self.state.no_edit_turns, self.state.inspection_recoveries), (0, 0))

    def test_forced_choice_rejection_keeps_notes_only_boundary(self):
        from src.tools import NOTES
        error = RuntimeError('tool_choice is not supported')
        error.status_code = 400
        agent = self.agent({'standard': [error, action('update_notes', findings=['calc.py:2 uses multiplier 1'])]})
        result = agent.request(self.state, 'standard', 'solver', [user_message('Record the next step')],
                               [NOTES], 1000, tool_choice={'type': 'function', 'function': {'name': 'update_notes'}})
        self.assertEqual(result['tool_calls'][0]['function']['name'], 'update_notes')
        self.assertEqual(self.client.requests[-1]['tool_choice'], 'auto')
        self.assertEqual([t['function']['name'] for t in self.client.requests[-1]['tools']], ['update_notes'])
        self.assertEqual(agent.budget.snapshot(0)['calls'], 2)

    def test_repair_context_survives_compaction(self):
        self.failed_fixture()
        self.state.update_notes({'repair_review': repair_review(), 'findings': ['verbose observation' * 70] * 24})
        context = compact_notebook(self.state, 7000)
        self.assertIn('previous_assumption', context)
        self.assertIn('previous_patch_behavior', context)
        self.assertIn('fix the positive example', context)

    def test_bad_repair_metadata_does_not_erase_prior_notes(self):
        self.failed_fixture()
        old = json.loads(json.dumps(self.state.notes))
        with self.assertRaises(ValueError):
            self.state.update_notes({'plan': [], 'scenarios': [], 'repair_review': 'A vague essay'})
        self.assertEqual(self.state.notes, old)
        with self.assertRaises(ValueError):
            self.state.update_notes({'next_step': None})

    def test_invalid_next_step_cannot_loop_until_entire_call_budget(self):
        cfg = replace(self.cfg, max_model_calls=40, inspection_turn_limit=3, inspection_recovery_turns=2,
                      stop_on_no_edit=True)
        self.agent(cfg=cfg, callback=lambda request: action('update_notes', plan=['Read the repository again'])).solve(self.task)
        self.assertEqual(len(self.client.requests), 7)

    def test_no_patch_does_not_end_task_after_the_old_seventy_five_turn_cutoff(self):
        cfg = replace(self.cfg, max_model_calls=100)
        idle = [action('update_notes', plan=['calc.py:triple replace the wrong multiplier'])] * 75
        self.agent({'standard': idle + [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(len(self.client.requests), 77)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        events = [json.loads(row) for row in (self.home / 'logs/0/events.jsonl').read_text().splitlines()]
        self.assertFalse(any(e['event'] == 'inspection_loop_stopped' for e in events))
        self.assertGreaterEqual(sum(e['event'] == 'inspection_checkpoint' for e in events), 3)

    def test_no_patch_still_respects_global_call_limit(self):
        cfg = replace(self.cfg, max_model_calls=9, inspection_turn_limit=2, inspection_recovery_turns=2)
        self.agent(cfg=cfg, callback=lambda request: action('update_notes', plan=['Still investigating'])).solve(self.task)
        self.assertEqual(len(self.client.requests), 9)
        events = [json.loads(row) for row in (self.home / 'logs/0/events.jsonl').read_text().splitlines()]
        self.assertFalse(any(e['event'] == 'inspection_loop_stopped' for e in events))
        self.assertEqual(events[-1]['reason'], 'task allowance reached')

    def test_stalled_starter_hands_actual_implementation_to_standard_and_passes(self):
        cfg = replace(self.cfg, solver_model='starter')
        idle = [action('update_notes', plan=['calc.py:triple replace x * 1 with x * 3'])] * 6
        self.agent({'starter': idle, 'standard': [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual([r['model'] for r in self.client.requests], ['starter'] * 6 + ['standard'] * 2)
        takeover = self.client.requests[6]
        self.assertIn('IMPLEMENTATION HANDOFF', json.dumps(takeover['messages']))
        self.assertEqual({t['function']['name'] for t in takeover['tools']},
                         {'bash', 'read_files', 'run_tests', 'update_notes', 'submit'})
        self.assertIn('return x * 1', json.dumps(takeover['messages']))
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)
        saved = json.loads((self.home / 'logs/0/working_state.json').read_text())
        self.assertEqual(saved['active_solver_model'], 'standard')

    def test_automatic_implementer_handoff_can_be_disabled(self):
        cfg = replace(self.cfg, solver_model='starter', solver_handoff_enabled=False)
        idle = [action('update_notes', plan=['Make the source edit'])] * 6
        self.agent({'starter': idle + [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertTrue(all(r['model'] == 'starter' for r in self.client.requests))
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_each_model_gets_its_own_output_cap_at_the_api_gateway(self):
        agent = self.agent({model: [completion('ok')] for model in ('starter', 'standard', 'expert')})
        for model in ('starter', 'standard', 'expert'):
            agent.request(self.state, model, 'fixture', [user_message('Hi')], [], 100000)
        self.assertEqual([r['max_tokens'] for r in self.client.requests], [8192, 16384, 8192])
        limited = replace(CONFIG, max_output_tokens=2000)
        self.assertEqual(limited.output_limit('standard'), 2000)

    def test_run_budget_allows_ten_dollars_but_rejects_above_it(self):
        self.assertEqual(CostBudget().run_limit, 10)
        budget = CostBudget(run_limit=10, task_limit=10, expert_limit=.4)
        with self.assertRaises(ValueError):
            CostBudget(run_limit=10.01)
        tickets = []
        for task in range(2):
            tickets.append(budget.reserve(task, 'standard', 'fixture', [], {}, 17000000, input_bound=1))
        self.assertLessEqual(sum(ticket.amount for ticket in tickets), 10)
        self.assertGreater(sum(ticket.amount for ticket in tickets), 9)

    def test_requested_high_output_caps_reach_gateway_when_affordable(self):
        cfg = replace(CONFIG, log_dir=self.cfg.log_dir, expert_budget_usd=.40)
        agent = self.agent({model: [completion('ok')] for model in ('standard', 'expert')}, cfg)
        for model in ('standard', 'expert'):
            agent.request(self.state, model, 'fixture', [user_message('Hi')], [], 100000)
        self.assertEqual([r['max_tokens'] for r in self.client.requests], [100000, 32768])

    def test_standard_model_headroom_is_separate_from_practical_solver_turn_cap(self):
        cfg = replace(self.cfg, standard_output_tokens=100000)
        self.agent({'standard': [self.edit(3), action('submit')]}, cfg).solve(self.task)
        self.assertEqual(cfg.output_limit('standard'), 100000)
        self.assertEqual(self.client.requests[0]['max_tokens'], cfg.solver_turn_output_tokens)
        prompt = self.client.requests[0]['messages'][0]['content']
        for instruction in ('PER-TURN ECONOMY', 'ceiling, not a target',
                            'little or no narration',
                            'stop the response once the next valid action',
                            'do not guess to appear fast'):
            self.assertIn(instruction, prompt)
        self.assertEqual(self.dispatcher.final['tests_failed'], 0)

    def test_high_expert_output_cap_respects_configured_small_dollar_limit(self):
        cfg = replace(CONFIG, log_dir=self.cfg.log_dir, expert_budget_usd=.18)
        agent = self.agent({'expert': [completion('ok')]}, cfg)
        agent.request(self.state, 'expert', 'fixture', [user_message('Hi')], [], 32768)
        actual = self.client.requests[0]['max_tokens']
        self.assertGreaterEqual(actual, 512)
        self.assertLess(actual, 18000)
        self.assertEqual(cfg.expert_budget_usd, .18)

    def test_issue_context_finds_implementation_outside_the_interface_file(self):
        target = self.repo / 'pkg/deep/resolve.py'
        target.parent.mkdir(parents=True)
        target.write_text('def resolve_author(author):\n    return None\n')
        subprocess.run(['git', 'add', 'pkg/deep/resolve.py'], cwd=self.repo, check=True, capture_output=True)
        self.state.task['problem_statement'] = 'Fix resolve_author(author) to use aliases.'
        self.state.task['interface'] = 'Path: calc.py; Name: triple(x)'
        navigation = self.tools.issue_context()
        self.assertIn('resolve_author', navigation['symbols'])
        self.assertTrue(any(r['path'] == 'pkg/deep/resolve.py' for r in navigation['source_requests']))
        self.assertTrue(any(hit['path'] == 'pkg/deep/resolve.py' and hit['line'] == 1
                            for hit in navigation['locations']))

    def test_counterexamples_allow_empty_structured_and_null_inputs(self):
        self.failed_fixture()
        review = repair_review()
        for value in ('', None, {'name': '', 'alternate_names': []}):
            review['new_scenarios'][0]['input'] = value
            review['new_scenarios'][0]['expected'] = None
            self.state.update_notes({'repair_review': review})
            self.assertEqual(self.state.repair_review['new_scenarios'][0]['input'], value)
        self.assertIsNone(self.state.repair_review['new_scenarios'][0]['expected'])

    def test_counterexample_identity_preserves_case_whitespace_and_types(self):
        keys = [TaskState.scenario_key({'input': value, 'expected': None})
                for value in ('JOHN', 'John', 'John ', '', None, 0, '0')]
        self.assertEqual(len(set(keys)), len(keys))

    def test_corrected_expectation_preserves_history_without_forcing_old_answer(self):
        agent, _ = self.failed_fixture()
        self.state.update_notes({'scenarios': ['Mistaken earlier expectation: -2 -> -4']})
        self.state.update_notes({'repair_review': repair_review('-2', '-6')})
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.state.update_notes({'review': 'The -4 expectation is superseded: the requirement says triple, '
                                'so -2 must yield -6. The current x * 3 code does so and preserves zero.'})
        self.ready(agent)
        self.assertIn('Mistaken earlier expectation: -2 -> -4', self.state.notes['scenarios'])
        self.assertEqual(agent.grade(self.state, self.tools, patch)['grading']['tests_failed'], 0)


if __name__ == "__main__":
    unittest.main()
