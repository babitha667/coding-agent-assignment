"""Regression checks for the last five-task run. No course API or paid grading."""

import json
import shlex
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_workflow as fixture
from src.task_state import TaskState
from src.time_policy import deadline_message, request_limits
from src.validation import classify, masks_status, test_failure_reason


class DeadlineCheckTests(unittest.TestCase):
    setUp = fixture.WorkflowTests.setUp
    agent = fixture.WorkflowTests.agent

    def perform(self, agent, name, **args):
        response = fixture.action(name, **args)['choices'][0]['message']
        results, requested = agent.execute_tools(self.state, self.tools, response, {name})
        return json.loads(results[0]['content']), requested

    def fixed(self):
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')

    def regression(self):
        (self.repo/'test_regression.py').write_text('import unittest\nfrom calc import triple\n'
            'class Example(unittest.TestCase):\n    def test_negative(self): self.assertEqual(triple(-2),-6)\n')

    def test_timeout_env_and_package_wrappers_are_recognized(self):
        for command in ('cd /app && timeout 300 go test ./scanner/ -run "TestX|TestY"',
                        'env -u TOKEN FOO=bar timeout --signal TERM --kill-after=5s 2m python3 -m pytest tests',
                        'gtimeout -k 5s 180 go test ./...', 'npm run test:unit -- --runInBand'):
            with self.subTest(command=command):
                self.assertTrue(classify(command)['test'])
        for command in ('echo "go test"', 'test -e foo', 'git grep "pytest"',
                        'python3 -c "print(\'go test\')"'):
            self.assertFalse(classify(command)['test'], command)

    def test_go_and_cargo_tests_provide_both_test_and_compile_evidence(self):
        for command in ('go test ./scanner/', 'cargo test -p library'):
            self.assertTrue(classify(command)['test'])
            self.assertTrue(classify(command)['build'])
        self.assertFalse(classify('yarn jest test_issue.ts')['build'])
        self.assertFalse(classify('mvn -version')['build'])
        self.assertFalse(classify('gradle --version')['build'])

    def test_build_kind_and_purpose_are_inferred_without_labels(self):
        for command in ('timeout 60 go build ./internal/config/', 'npm run build',
                        './node_modules/.bin/tsc --project tsconfig.json --noEmit'):
            with self.subTest(command=command), patch.object(self.tools.sb, 'exec', return_value={
                    'output': '', 'exit_code': 0, 'timed_out': False}):
                result = self.tools.bash({'command': command})
                self.assertEqual(result['purpose'], 'check')
                self.assertEqual(result['check_kind'], 'build')

    def test_explicit_kind_without_purpose_does_not_reject_real_check(self):
        result = self.tools.bash({'command': 'python3 -c "assert 2+2 == 4"', 'check_kind': 'smoke'})
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['purpose'], 'check')

    def test_gocheck_skipped_wrapper_is_not_passing_behavior(self):
        output = ('=== RUN   TestDynamoevents\nOK: 0 passed, 1 skipped\n'
                  '--- PASS: TestDynamoevents (0.00s)\nPASS\nok example\n')
        self.assertIn('All selected tests were skipped', test_failure_reason({'output': output}))
        mixed = output + '=== RUN   TestRealUnit\n--- PASS: TestRealUnit (0.00s)\n'
        self.assertEqual(test_failure_reason({'output': mixed}), '')

    def test_empty_go_package_is_not_behavioral_validation(self):
        self.assertIn('No tests executed', test_failure_reason({'output': '? example [no test files]\n'}))
        self.assertIn('No tests executed', test_failure_reason({'output': 'ok example 0.1s [no tests to run]\n'}))
        self.assertEqual(test_failure_reason({'output': '? empty [no test files]\nok tested/package 0.2s\n'}), '')

    def test_successful_and_failing_real_counts_are_not_skipped(self):
        for output in ('2 passed, 1 skipped in 0.1s', '1 failed, 2 skipped in 0.1s',
                       'ERROR importing dependency', '--- PASS: TestValue\nPASS\nok example'):
            self.assertEqual(test_failure_reason({'output': output}), '', output)

    def test_check_error_cannot_be_hidden_by_following_echo(self):
        result = self.tools.bash({'command': 'python3 -c "assert False"; echo "EXIT:$?"', 'purpose': 'test'})
        self.assertNotEqual(result['exit_code'], 0)
        self.assertNotIn('EXIT:', result['output'])

    def test_or_masked_check_is_rejected_before_execution(self):
        self.assertFalse(masks_status('node -e "const x = 0 || 1; if (x !== 1) throw Error()"'))
        self.assertFalse(masks_status("node <<'JS'\nconst x = 0 || 1;\nJS"))
        with patch.object(self.tools.sb, 'exec') as execute:
            result = self.tools.bash({'command': 'go test ./scanner || echo ignored'})
            self.assertIn('masks failure', result['error'])
            execute.assert_not_called()

    def test_tsc_source_arguments_are_rejected_when_project_config_exists(self):
        self.tools.validation_metadata = {'typescript': {'project': 'tsconfig.json'}}
        with patch.object(self.tools.sb, 'exec') as execute:
            result = self.tools.bash({'command': 'npx tsc --noEmit src/issue.ts'})
            self.assertIn('ignores its settings', result['error'])
            execute.assert_not_called()

    def test_project_settings_are_given_before_solving(self):
        (self.repo/'package.json').write_text(json.dumps({'scripts': {'test': 'jest', 'typecheck': 'tsc -p tsconfig.json',
                                                                    'build': 'webpack', 'start': 'serve'}}))
        (self.repo/'tsconfig.json').write_text('{}')
        (self.repo/'yarn.lock').write_text('')
        info = self.tools.validation_context()
        self.assertEqual(info['package_manager'], 'yarn')
        self.assertEqual(info['scripts']['typecheck'], 'tsc -p tsconfig.json')
        self.assertNotIn('start', info['scripts'])
        self.assertIn('--project tsconfig.json', info['typescript']['focused_typecheck'])

    def test_blocked_submit_returns_actual_gate_not_grading_requested(self):
        agent = self.agent()
        self.fixed()
        result, requested = self.perform(agent, 'submit')
        self.assertTrue(requested)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('error', result)
        self.assertNotIn('grading requested', json.dumps(result))
        self.assertEqual(self.dispatcher.results, [])

    def test_submit_returns_completed_grading_and_records_real_request(self):
        agent = self.agent()
        self.fixed()
        self.regression()
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        result, requested = self.perform(agent, 'submit')
        self.assertEqual(result['status'], 'graded')
        self.assertEqual(result['grading']['tests_failed'], 0)
        self.assertEqual(len(self.dispatcher.results), 1)
        events = [json.loads(x) for x in (self.state.directory/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(x['event'] == 'evaluation_requested' for x in events), 1)

    def test_blocked_submit_needs_another_explicit_submit_after_checks(self):
        agent = self.agent()
        self.fixed()
        self.perform(agent, 'submit')
        self.regression()
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        self.assertEqual(self.dispatcher.results, [])
        result, _ = self.perform(agent, 'submit')
        self.assertEqual(result['grading']['tests_failed'], 0)

    def test_edit_after_blocked_submit_does_not_queue_evaluation(self):
        agent = self.agent()
        (self.repo/'calc.py').write_text('def triple(x):\n    return abs(x)*3\n')
        self.perform(agent, 'submit')
        self.fixed()
        self.regression()
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        self.assertEqual(self.dispatcher.results, [])

    def test_timer_and_passing_tests_without_submit_intent_do_not_grade(self):
        agent = self.agent()
        self.fixed()
        self.regression()
        self.state.started -= 1300
        self.assertIsNotNone(deadline_message(self.state))
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        self.assertEqual(self.dispatcher.results, [])

    def test_verified_candidate_waits_for_submit_and_duplicate_is_blocked(self):
        agent = self.agent()
        self.fixed()
        self.regression()
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        self.assertEqual(self.dispatcher.results, [])
        result, _ = self.perform(agent, 'submit')
        self.assertEqual(result['grading']['tests_failed'], 0)
        self.assertEqual(len(self.dispatcher.results), 1)
        duplicate, _ = self.perform(agent, 'submit')
        self.assertEqual(duplicate['status'], 'blocked')
        self.assertEqual(len(self.dispatcher.results), 1)

    def test_solve_waits_for_model_submit_after_successful_checks(self):
        cfg = replace(self.cfg, review_enabled=False)
        self.fixed()
        self.regression()
        actions = [fixture.action('run_tests', command='python3 -B -m unittest test_regression'),
                   fixture.action('submit')]
        def callback(request):
            self.assertEqual(self.dispatcher.results, [])
            return actions.pop(0)
        agent = self.agent(cfg=cfg, callback=callback)
        result = agent.solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(len(self.dispatcher.results), 1)

    def test_explicit_submit_does_not_bypass_failed_or_missing_tests(self):
        agent = self.agent()
        self.fixed()
        self.perform(agent, 'bash', command="python3 -c 'from calc import triple; triple(2)'",
                                purpose='check', check_kind='smoke', covers=['calc.py'])
        feedback, _ = self.perform(agent, 'submit')
        self.assertIn('error', feedback)
        self.assertEqual(self.dispatcher.results, [])
        self.regression()
        (self.repo/'calc.py').write_text('def triple(x):\n    return abs(x)*3\n')
        self.perform(agent, 'run_tests', command='python3 -B -m unittest test_regression')
        result, _ = self.perform(agent, 'submit')
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(self.dispatcher.results, [])

    def test_allowance_ending_does_not_auto_grade_a_tested_unsubmitted_patch(self):
        cfg = replace(self.cfg, max_model_calls=1, review_enabled=False)
        self.fixed()
        self.regression()
        agent = self.agent({'standard': [fixture.action('run_tests',
                           command='python3 -B -m unittest test_regression')]}, cfg)
        agent.solve(self.task)
        self.assertEqual(self.dispatcher.results, [])

    def test_solver_prompt_prioritizes_implementation_and_explicit_submit(self):
        from src.prompts import SOLVER_PROMPT
        self.assertIn('PRIORITY: Deliver the required implementation', SOLVER_PROMPT)
        self.assertIn('MAKE that edit', SOLVER_PROMPT)
        self.assertIn('Only YOUR explicit submit call starts grading', SOLVER_PROMPT)
        self.assertIn('call submit AGAIN', SOLVER_PROMPT)

    def test_expert_dollar_allowance_falls_back_to_standard_with_same_evidence(self):
        cfg = replace(self.cfg, expert_budget_usd=.00001)
        agent = self.agent({'standard': [fixture.completion(fixture.WorkflowTests.report())]}, cfg)
        self.tools.read_files({'path': 'calc.py'})
        self.state.record_grade('candidate', {'tests_failed': 1, 'tests_total': 8, 'attempt': 0})
        messages = []
        agent.maybe_review(self.state, self.tools, 'candidate', messages)
        self.assertEqual([r['model'] for r in self.client.requests], ['standard'])
        self.assertIn('calc.py', json.dumps(self.client.requests[0]['messages']))
        self.assertEqual(self.state.review_history[-1]['status'], 'accepted')
        self.assertTrue(self.state.review_history[-1]['fallback_reason'])
        agent.maybe_review(self.state, self.tools, 'candidate', messages)
        self.assertEqual(len(self.client.requests), 1)

    def test_fallback_does_not_override_shared_task_funds_or_repair_reserve(self):
        cfg = replace(self.cfg, expert_budget_usd=.00001, task_budget_usd=.02)
        agent = self.agent(cfg=cfg)
        self.state.record_grade('candidate', {'tests_failed': 1, 'tests_total': 8, 'attempt': 0})
        agent.maybe_review(self.state, self.tools, 'candidate', [])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.state.review_history[-1]['status'], 'unavailable')

    def test_solver_turn_cap_and_late_cap_do_not_reduce_reviewer_limit(self):
        self.assertEqual(request_limits(self.state, 'solver', 100000)[0], 32768)
        self.assertEqual(request_limits(self.state, 'review_1', 32768)[0], 32768)
        self.state.started -= 1300
        cap, timeout = request_limits(self.state, 'solver', 100000)
        self.assertEqual(cap, 8192)
        self.assertLessEqual(timeout, 60)

    def test_deadline_checkpoints_are_phase_specific_not_every_turn(self):
        self.state.started -= 500
        self.assertIn('TIME CHECKPOINT', deadline_message(self.state))
        self.assertIsNone(deadline_message(self.state))
        self.state.started -= 800
        self.assertIn('TIME CHECKPOINT', deadline_message(self.state))
        self.assertIsNone(deadline_message(self.state))

    def test_whitespace_does_not_discard_runnable_patch_but_syntax_still_blocks(self):
        from src.code_checks import static_checks
        agent = self.agent()
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3  \n')
        agent.current_patch(self.state)
        checks = static_checks(self.tools.sb)
        self.assertTrue(checks['ok'], checks)
        self.assertIn('whitespace', str(checks['warnings']))
        (self.repo/'calc.py').write_text('def triple(:\n')
        agent.current_patch(self.state)
        self.assertFalse(static_checks(self.tools.sb)['ok'])

    def test_go_test_records_both_checks_in_agent_without_labels(self):
        agent = self.agent()
        self.fixed()
        execute = self.tools.sb.exec
        def fake_go(command, timeout_s, **kwargs):
            if command.startswith('bash ') and 'go test' in command:
                return {'output': '=== RUN TestReal\n--- PASS: TestReal\nPASS\nok example\n',
                        'exit_code': 0, 'timed_out': False}
            return execute(command, timeout_s, **kwargs)
        with patch.object(self.tools.sb, 'exec', side_effect=fake_go):
            result, _ = self.perform(agent, 'bash', command='timeout 60 go test ./scanner/')
        self.assertEqual(result['purpose'], 'test')
        self.assertEqual({r['kind'] for r in self.state.readiness_checks}, {'smoke', 'build'})
        self.assertTrue(self.state.last_local_test['passed'])

    def test_full_loop_blocked_submit_requires_explicit_resubmission(self):
        def write(path, text):
            return fixture.action('bash', command='python3 -c ' + shlex.quote(
                f'from pathlib import Path; Path({path!r}).write_text({text!r})'))
        test = ('import unittest\nfrom calc import triple\nclass Example(unittest.TestCase):\n'
                '    def test_negative(self): self.assertEqual(triple(-2), -6)\n')
        scripts = {'standard': [write('calc.py', 'def triple(x):\n    return x * 3\n'),
            fixture.action('submit'), write('test_regression.py', test),
            fixture.action('run_tests', command='python3 -B -m unittest test_regression'),
            fixture.action('submit')]}
        result = self.agent(scripts).solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        self.assertEqual(len(self.client.requests), 5)
        self.assertEqual(len(self.dispatcher.results), 1)
        messages = self.client.requests[2]['messages']
        blocked = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        self.assertTrue(any(m.get('status') == 'blocked' and m.get('error') for m in blocked))

    def test_short_task_experiments_keep_proportional_time_allowances(self):
        self.state.config = replace(self.cfg, task_timeout_s=60)
        cap, timeout = request_limits(self.state, 'solver', 100000)
        self.assertEqual(cap, 32768)
        self.assertGreater(timeout, 30)
        self.assertIsNone(deadline_message(self.state))

    def test_api_slot_wait_refreshes_deadline_context_and_request_limits(self):
        agent = self.agent({'standard': [fixture.completion('next action')]})
        state = self.state
        class WaitedSlot:
            def acquire(self, timeout):
                state.started -= 1300
                return True
            def release(self):
                pass
        agent.api_slots = WaitedSlot()
        agent.request(state, 'standard', 'solver', [{'role': 'user', 'content': 'Continue the fix'}], [], 100000)
        request = self.client.requests[0]
        self.assertEqual(request['max_tokens'], 8192)
        self.assertLessEqual(request['timeout'], 60)
        self.assertTrue(any('TIME CHECKPOINT' in m['content'] for m in request['messages']))


if __name__ == '__main__':
    unittest.main()
