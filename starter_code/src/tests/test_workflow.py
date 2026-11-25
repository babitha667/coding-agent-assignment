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


class LocalSandbox:
    def __init__(self, url, workdir):
        self.workdir = str(workdir)

    def exec(self, command, timeout_s, **kwargs):
        try:
            result = subprocess.run(command, shell=True, cwd=self.workdir, capture_output=True,
                                    text=True, timeout=timeout_s)
            return {'output': result.stdout + result.stderr, 'exit_code': result.returncode, 'timed_out': False}
        except subprocess.TimeoutExpired:
            return {'output': 'fixture command timeout', 'exit_code': 124, 'timed_out': True}


class NullLogger:
    def __getattr__(self, key):
        return lambda *args, **kwargs: None


def completion(content='', calls=(), finish=None, usage=None):
    tools = [{'id': f'call_{i}', 'type': 'function', 'function': {'name': name,
              'arguments': json.dumps(args)}} for i, (name, args) in enumerate(calls)]
    return {'choices': [{'finish_reason': finish or ('tool_calls' if tools else 'stop'),
                         'message': {'content': content, 'tool_calls': tools}}],
            'usage': usage or {'prompt_tokens': 500, 'completion_tokens': 100, 'total_tokens': 600}}


def action(name, **args):
    return completion(calls=[(name, args)])


def repair_review(case_input='-2', expected='-6'):
    return {'previous_assumption': 'The earlier examples covered the intended multiplier.',
            'missed_behavior': 'Negative inputs also require multiplying by three.',
            'new_scenarios': [{'input': case_input, 'expected': expected,
                'previous_patch_behavior': 'The previous x * 2 implementation gives -4 for -2.',
                'requirement': 'The task explicitly requires multiplying by three, including negative x.'}],
            'planned_changes': ['calc.py:triple: replace the incorrect multiplier with 3.']}


def record_repair():
    return action('update_notes', repair_review=repair_review())


def review_patch():
    return action('update_notes', review='Reviewed current calc.py:triple against the preserved examples '
                  'and the new -2 -> -6 scenario; checked zero, signs and the unchanged signature. '
                  'This is reference reasoning, not an executed test result.')


class ScriptedClient:
    def __init__(self, scripts=None, callback=None):
        self.scripts = {key: list(value) for key, value in (scripts or {}).items()}
        self.callback, self.requests = callback, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **request):
        self.requests.append(json.loads(json.dumps(request)))
        value = self.callback(request) if self.callback else self.scripts[request['model']].pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeDispatcher:
    """Matches extract's integer return and done's evaluated/continue behavior."""
    def __init__(self, root, grades=()):
        self.root, self.grade_script = Path(root), list(grades)
        self.patchfile = self.root.parent / ('submission-' + self.root.name + '.patch')
        self.evaluated = False
        self.results, self.submissions = [], []
        self.continues, self.final = 0, None

    def logger(self, key):
        return NullLogger()

    def patch_path(self, key):
        return str(self.patchfile)

    def extract_patch(self, key):
        subprocess.run(['git', 'add', '-N', '.'], cwd=self.root, check=True, capture_output=True)
        patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=self.root).decode()
        self.patchfile.write_text(patch)
        return len(patch.encode())

    def submit_patch(self, key, patch):
        self.patchfile.write_text(patch)
        self.submissions.append(patch)

    def evaluate(self, key):
        self.assert_patch = self.patchfile.read_text()
        if self.evaluated:
            raise RuntimeError('fixture: continue_task required before regrading')
        failed = self.grade_script.pop(0) if self.grade_script else int('+    return x * 3' not in self.assert_patch)
        result = {'tests_failed': failed, 'tests_total': 8, 'attempt': self.continues}
        self.results.append(result)
        self.evaluated = True
        return result

    def continue_task(self, key):
        self.continues += 1
        self.evaluated = False
        self.patchfile.write_text('')

    def done(self, key, **kwargs):
        if not self.evaluated and self.patchfile.read_text().strip():
            raise AssertionError('Uncounted done auto-evaluation')
        self.final = self.results[-1] if self.results else {'tests_failed': 8, 'tests_total': 8}
        return self.final


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='notebook-agent-')
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.repo = self.home / 'repo'
        self.repo.mkdir()
        self.cfg = replace(CONFIG, workers=1, max_model_calls=30, review_min_remaining_s=1,
                           standard_output_tokens=16384, expert_output_tokens=8192,
                           expert_review_output_tokens=8192, review_format_retries=1,
                           log_dir=str(self.home / 'logs'))
        (self.repo / 'calc.py').write_text('def triple(x):\n    return x * 1\n')
        (self.repo / 'README.md').write_text('A tiny arithmetic library. triple should multiply by three.\n')
        for command in (['git', 'init', '-q'], ['git', 'add', '.'], ['git', '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'baseline']):
            subprocess.run(command, cwd=self.repo, check=True, capture_output=True)
        self.task = {'task': 0, 'sandbox_url': 'offline', 'workdir': str(self.repo),
                     'problem_statement': 'Fix triple(x); multiply by 3, including zero and negative x.',
                     'requirements': 'Preserve the function signature.', 'interface': 'triple(x)'}
        self.dispatcher = FakeDispatcher(self.repo)
        self.state = TaskState(self.task, self.cfg)
        self.state.role_calls = {}
        self.tools = TaskTools(LocalSandbox('offline', self.repo), self.state, self.cfg)

    def agent(self, scripts=None, cfg=None, callback=None):
        self.client = ScriptedClient(scripts, callback)
        return Agent(self.dispatcher, cfg or self.cfg, self.client, LocalSandbox)

    def edit(self, factor):
        return completion(calls=[('bash', {
            'command': rf"sed -i.bak 's/x \* [0-9]/x * {factor}/' calc.py && rm calc.py.bak", 'purpose': 'edit'}),
            ('bash', self.smoke_arguments())])

    @staticmethod
    def smoke_arguments(command="python3 -B -c 'from calc import triple; triple(0)'", covers=None, kind='smoke'):
        return {'command': command, 'purpose': 'check', 'check_kind': kind, 'covers': covers or ['calc.py']}

    def ready(self, agent, state=None, tools=None):
        message = completion(calls=[('bash', self.smoke_arguments())])['choices'][0]['message']
        return agent.execute_tools(state or self.state, tools or self.tools, message, {'bash'})

    @staticmethod
    def report():
        return ('VERDICT: REPAIR\nNEXT_ACTION: EDIT calc.py:triple to preserve negative signs.\n'
                'PREVIOUS_CYCLE_GAP: Positive-only tests missed negative input.\n'
                'TYPE: CONFIRMED_DEFECT\nLOCATION: calc.py:triple\n'
                'REQUIREMENT: Multiply every input by three.\n'
                'EVIDENCE: abs(x) removes the sign before multiplication.\n'
                'CASE: -2 must produce -6, but the current branch produces 6.\n'
                'ACTION: Replace abs(x) * 3 with x * 3.\n'
                'VERIFY: Add a negative regression and preserve positive/zero tests.')

    def failed_candidate(self, patch='candidate one'):
        self.state.observe_patch(patch)
        self.state.record_grade(patch, {'tests_failed': 2, 'tests_total': 8, 'attempt': 0})

    def test_production_defaults_keep_models_caps_and_budgets(self):
        self.assertEqual(CONFIG.verification_mode, 'local_tests')
        self.assertEqual((CONFIG.solver_model, CONFIG.reviewer_model), ('standard', 'expert'))
        self.assertEqual((CONFIG.standard_output_tokens, CONFIG.expert_output_tokens), (100000, 32768))
        self.assertEqual((CONFIG.task_budget_usd, CONFIG.run_budget_usd), (.50, 10))
        self.assertEqual((CONFIG.expert_budget_usd, CONFIG.expert_calls), (.40, 9))

    def test_no_pregrading_reviews_even_when_forced_or_stalled(self):
        agent = self.agent()
        self.state.started -= 600
        self.state.no_edit_turns = self.state.stalled_turns = 200
        for patch in ('', 'ungraded source patch'):
            for force in (False, True):
                agent.maybe_review(self.state, self.tools, patch, [], force=force)
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.state.review_rounds, 0)

    def test_one_review_for_each_fresh_failed_grade_not_each_solver_turn(self):
        cfg = replace(self.cfg, expert_review_calls=1)
        agent = self.agent({'expert': [completion(self.report()), completion(self.report())]}, cfg)
        for i in range(2):
            patch = f'candidate {i}'
            self.state.observe_patch(patch)
            self.state.record_grade(patch, {'tests_failed': 2-i, 'tests_total': 8, 'attempt': i})
            for _ in range(5):
                self.state.role_calls['solver'] = self.state.role_calls.get('solver', 0)+100
                agent.maybe_review(self.state, self.tools, patch, [], force=True)
        self.assertEqual(len(self.client.requests), 2)
        self.assertTrue(all(r['model'] == 'expert' for r in self.client.requests))

    def test_passed_grade_does_not_trigger_review(self):
        agent = self.agent()
        self.state.record_grade('good', {'tests_failed': 0, 'tests_total': 8, 'attempt': 0})
        agent.maybe_review(self.state, self.tools, 'good', [], force=True)
        self.assertEqual(self.client.requests, [])

    def test_expert_reads_then_fresh_report_request_has_no_tools_or_tool_history(self):
        self.state.update_notes({'findings': ['Earlier round diagnosis: inspect signed input handling']})
        self.failed_candidate()
        evidence = completion(calls=[('read_round', {'attempt': 0, 'section': 'diagnosis'}),
                                     ('read_files', {'requests': [{'path': 'calc.py'}]})])
        agent = self.agent({'expert': [evidence, completion(self.report())]})
        messages = []
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        first, final = self.client.requests
        self.assertIn('tools', first)
        self.assertNotIn('tools', final)
        self.assertNotIn('tool_choice', final)
        self.assertFalse(any(m['role'] in ('tool', 'assistant') for m in final['messages']))
        self.assertIn('return x * 1', json.dumps(final['messages']))
        self.assertIn('Earlier round diagnosis', json.dumps(final['messages']))
        self.assertIn(self.report(), messages[-1]['content'])
        self.assertEqual(self.state.review_history[-1]['status'], 'accepted')

    def test_expert_can_finish_without_unnecessary_source_read(self):
        self.failed_candidate()
        agent = self.agent({'expert': [completion(self.report())]})
        agent.maybe_review(self.state, self.tools, 'candidate one', [])
        self.assertEqual(len(self.client.requests), 1)

    def test_printed_tool_markup_gets_one_accounted_tool_free_format_correction(self):
        self.failed_candidate()
        bad = 'VERDICT: REPAIR\nNEXT_ACTION: EDIT calc.py\n<｜DSML｜tool_calls>read_files</｜DSML｜tool_calls>'
        agent = self.agent({'expert': [action('read_files', requests=[{'path': 'calc.py'}]),
                                      completion(bad), completion(self.report())]})
        messages = []
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        self.assertEqual(len(self.client.requests), 3)
        self.assertEqual(self.budget_calls(agent), 3)
        self.assertNotIn('<｜DSML｜', messages[-1]['content'])
        self.assertIn(self.report(), messages[-1]['content'])

    @staticmethod
    def budget_calls(agent):
        return agent.budget.snapshot(0)['expert_calls']

    def test_invalid_reviews_do_not_count_as_success_or_repeat_for_same_grade(self):
        self.failed_candidate()
        cfg = replace(self.cfg, expert_review_calls=1)
        agent = self.agent({'expert': [completion('<tool_call>read_files</tool_call>')]*2}, cfg)
        messages = []
        for _ in range(3):
            agent.maybe_review(self.state, self.tools, 'candidate one', messages, force=True)
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.state.review_history[-1]['status'], 'unavailable')
        self.assertEqual(self.state.review_history[-1]['advice'], '')
        self.assertIn('no advice was accepted', messages[-1]['content'])

    def test_review_source_failure_does_not_crash_the_solver_or_repeat_the_review(self):
        self.failed_candidate()
        self.state.record_read('calc.py', 'hash', 1, 2, 'current source')
        agent = self.agent()
        def unavailable(*args, **kwargs):
            raise ConnectionError('sandbox read temporarily unavailable')
        self.tools.read_files = unavailable
        messages = []
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        self.assertEqual(self.state.review_history[-1]['status'], 'unavailable')
        self.assertEqual(self.state.review_rounds, 1)
        self.assertEqual(self.client.requests, [])
        self.assertIn('sandbox read temporarily unavailable', messages[-1]['content'])

    def test_needs_evidence_is_precise_and_not_an_unsupported_edit(self):
        self.failed_candidate()
        report = ('VERDICT: NEEDS_EVIDENCE\nNEXT_ACTION: INSPECT calc.py:triple caller for signed input contract.\n'
                  'EVIDENCE: Caller behavior is absent; whether negatives are accepted changes the repair.')
        agent = self.agent({'expert': [completion(report)]})
        messages = []
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        self.assertIn(report, messages[-1]['content'])
        self.assertNotIn('EDIT ', report)

    def test_report_rejects_unsubstantiated_wrong_action_or_incomplete_advice(self):
        for report in ('Do more reading', 'VERDICT: REPAIR\nNEXT_ACTION: EDIT calc.py',
                       'VERDICT: NEEDS_EVIDENCE\nNEXT_ACTION: EDIT calc.py\nEVIDENCE: Unknown',
                       'VERDICT: REPAIR\nNEXT_ACTION: EDIT calc.py\n<think>unfinished'):
            with self.subTest(report=report), self.assertRaises(ValueError):
                final_review_report(report)

    def test_complete_final_report_after_deliberation_is_preserved_losslessly(self):
        report = self.report() + '\n' + 'Evidence detail. '*1500
        self.assertEqual(final_review_report('<think>untrusted preamble</think>'+report), report.strip())

    def test_recent_helper_evidence_and_repository_inventory_reach_reviewer(self):
        self.state.record_read('catalog/utils.py', 'hash', 41, 63, 'def author_dates_match: uses year extraction')
        packet = self.agent().review_packet(self.state, {'files': []}, 'patch', 'failed grading')
        self.assertIn('uses year extraction', packet)
        self.assertIn('Repo structure', packet)
        self.assertIn('Inspected source inventory', packet)

    def test_malformed_whole_notebook_in_findings_rejected_without_losing_plan(self):
        self.state.update_notes({'plan': ['Preserve a valid plan']})
        before = json.dumps(self.state.notes, sort_keys=True)
        with self.assertRaises(ValueError):
            self.state.update_notes({'plan': ['Would overwrite'], 'findings': '["fact"], "plan": ["hidden"]'})
        self.assertEqual(json.dumps(self.state.notes, sort_keys=True), before)

    def test_notes_accept_valid_json_array_but_not_objects_or_unknown_fields(self):
        self.state.update_notes({'findings': '["calc.py:2 wrong multiplier"]'})
        self.assertEqual(self.state.notes['findings'], ['calc.py:2 wrong multiplier'])
        for bad in ({'plan': {'steps': ['edit']}}, {'plan': [None]}, {'unknown': 'oops'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.state.update_notes(bad)

    def test_local_test_scope_is_automatic_but_printed_pass_is_not_validation(self):
        result = self.tools.bash({'command': 'python3 -c "assert True"', 'purpose': 'test'})
        self.assertEqual(result['purpose'], 'test')
        self.assertIn('covers', result)
        with self.assertRaises(ValueError):
            self.tools.bash({'command': 'echo PASS', 'purpose': 'test', 'covers': ['calc.py']})

    def test_grade_is_blocked_without_tests_and_after_edit_invalidates_pass(self):
        agent = self.agent()
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.ready(agent)
        self.assertIn('local tests', agent.grade(self.state, self.tools, patch)['error'])
        call = action('bash', command="python3 -B -c 'from calc import triple; assert triple(-2)==-6'",
                      purpose='test', covers=['calc.py'])['choices'][0]['message']
        agent.execute_tools(self.state, self.tools, call, {'bash'})
        self.assertTrue(self.state.last_local_test['passed'])
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 2\n')
        agent.current_patch(self.state)
        self.assertIsNone(self.state.last_local_test)
        self.assertEqual(self.dispatcher.results, [])

    def test_test_failure_record_includes_assertion_output_for_review(self):
        agent = self.agent()
        call = action('bash', command="python3 -B -c 'from calc import triple; assert triple(2)==6'",
                      purpose='test', covers=['calc.py'])['choices'][0]['message']
        agent.execute_tools(self.state, self.tools, call, {'bash'})
        self.assertFalse(self.state.last_local_test['passed'])
        self.assertIn('AssertionError', self.state.last_local_test['output'])
        packet = agent.review_packet(self.state, {'files': []}, 'patch', 'failed grading')
        self.assertIn('AssertionError', packet)

    def test_latest_failed_test_overrides_prior_passing_test(self):
        agent = self.agent()
        for expected in (2, 6):
            msg = action('bash', command=f"python3 -B -c 'from calc import triple; assert triple(2)=={expected}'",
                         purpose='test', covers=['calc.py'])['choices'][0]['message']
            agent.execute_tools(self.state, self.tools, msg, {'bash'})
        self.assertFalse(self.state.last_local_test['passed'])
        self.assertEqual(len(self.state.test_history), 2)

    def test_grade_rejects_test_coverage_that_omits_another_changed_source(self):
        cfg = replace(self.cfg, require_code_readiness=False)
        agent = self.agent(cfg=cfg)
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        (self.repo/'other.py').write_text('def helper():\n    return 42\n')
        call = action('bash', command="python3 -B -c 'from calc import triple; assert triple(2)==6'",
                      purpose='test', covers=['calc.py'])['choices'][0]['message']
        agent.execute_tools(self.state, self.tools, call, {'bash'})
        feedback = agent.grade(self.state, self.tools, self.state.patch_text)
        self.assertEqual(feedback['missing_test_coverage'], ['other.py'])
        self.assertEqual(self.dispatcher.results, [])

    def test_separate_test_batches_combine_coverage_only_on_the_same_patch(self):
        from src.code_checks import local_test_gaps
        self.state.observe_patch('current candidate')
        self.state.test_history = [
            {'patch_sha256': self.state.patch_hash, 'covers': ['a.py'], 'passed': True},
            {'patch_sha256': self.state.patch_hash, 'covers': ['b.py'], 'passed': True}]
        self.assertEqual(local_test_gaps(self.state, ['a.py', 'b.py']), [])
        self.state.test_history.append({'patch_sha256': self.state.patch_hash, 'covers': ['a.py'], 'passed': False})
        self.assertEqual(local_test_gaps(self.state, ['a.py', 'b.py']), ['a.py'])
        self.state.observe_patch('edited candidate')
        self.assertEqual(local_test_gaps(self.state, ['a.py', 'b.py']), ['a.py', 'b.py'])

    def test_review_model_remains_configurable_without_adding_a_stall_reviewer(self):
        cfg = replace(self.cfg, reviewer_model='standard', review_calls=1)
        self.failed_candidate()
        agent = self.agent({'standard': [completion(self.report())]}, cfg)
        agent.maybe_review(self.state, self.tools, 'candidate one', [])
        self.assertEqual(self.client.requests[0]['model'], 'standard')
        self.assertNotIn('tools', self.client.requests[0])

    def test_truncated_final_review_is_discarded_and_correction_is_budgeted(self):
        self.failed_candidate()
        cfg = replace(self.cfg, expert_review_calls=1)
        agent = self.agent({'expert': [completion(self.report(), finish='length'), completion(self.report())]}, cfg)
        agent.maybe_review(self.state, self.tools, 'candidate one', [])
        self.assertEqual(self.state.review_history[-1]['physical_calls'], 2)
        self.assertEqual(self.state.review_history[-1]['status'], 'accepted')
        self.assertEqual(self.budget_calls(agent), 2)

    def test_full_accepted_review_survives_compaction_without_replacing_solver_assessment(self):
        self.failed_candidate()
        cfg = replace(self.cfg, expert_review_calls=1, context_max_chars=8000)
        report = self.report() + '\nDETAIL: ' + 'source evidence '*1000
        agent = self.agent({'expert': [completion(report)]}, cfg)
        messages = [{'role': 'system', 'content': 'instructions'}, {'role': 'user', 'content': 'issue'}]
        agent.maybe_review(self.state, self.tools, 'candidate one', messages)
        self.state.update_notes({'review': 'Solver assessment, separate from Expert report.'})
        compacted = agent.compact(self.state, messages)
        self.assertTrue(any(report.strip() in m.get('content', '') for m in compacted))
        self.assertEqual(self.state.notes['review'], 'Solver assessment, separate from Expert report.')

    def test_no_timer_submission_while_solver_is_still_working(self):
        responses = [self.edit(3), action('read_files', requests=[{'path': 'calc.py'}])]
        def callback(request):
            self.assertEqual(self.dispatcher.results, [])
            if responses:
                return responses.pop(0)
            return action('submit')
        cfg = replace(self.cfg, verification_mode='reference', review_enabled=False, max_model_calls=3)
        result = self.agent(cfg=cfg, callback=callback).solve(self.task)
        self.assertEqual(result['tests_failed'], 0)

    def test_expert_exhaustion_uses_logged_standard_fallback(self):
        cfg = replace(self.cfg, expert_calls=1, expert_review_calls=1, review_format_retries=0)
        agent = self.agent({'expert': [completion(self.report())],
                            'standard': [completion(self.report())]}, cfg)
        for i in range(2):
            patch = f'patch {i}'
            self.state.observe_patch(patch)
            self.state.record_grade(patch, {'tests_failed': 2, 'tests_total': 8, 'attempt': i})
            agent.maybe_review(self.state, self.tools, patch, [])
        self.assertEqual([r['model'] for r in self.client.requests], ['expert', 'standard'])
        self.assertEqual(self.state.review_history[-1]['model'], 'standard')
        self.assertIn('Expert-call allowance', self.state.review_history[-1]['fallback_reason'])

    def test_end_to_end_red_green_grade_expert_repair_and_regrade(self):
        regression = ('import unittest\nfrom calc import triple\nclass Regression(unittest.TestCase):\n'
                      '    def test_positive(self): self.assertEqual(triple(2),6)\n'
                      '    def test_zero(self): self.assertEqual(triple(0),0)\n')
        def write(path, text):
            return action('bash', command='python3 -c '+__import__('shlex').quote(
                f'from pathlib import Path; Path({path!r}).write_text({text!r})'), purpose='edit')
        def tests():
            return action('bash', command='python3 -B -m unittest test_regression -v',
                          purpose='test', covers=['calc.py', 'test_regression.py'])
        self.dispatcher.grade_script = [1, 0]
        record = repair_review()
        record['previous_assumption'] = 'Positive and zero input were sufficient.'
        record['new_scenarios'][0]['previous_patch_behavior'] = 'abs(-2)*3 produces 6, not -6.'
        scripts = {'standard': [action('read_files', requests=[{'path':'calc.py'}]),
            action('update_notes', findings=['calc.py uses x*1 instead of x*3'],
                   plan=['Write positive/zero regression, fix multiplier'], scenarios=['2 -> 6; 0 -> 0']),
            write('test_regression.py', regression), tests(),
            write('calc.py', 'def triple(x):\n    return abs(x) * 3\n'), tests(), action('submit'),
            action('update_notes', repair_review=record),
            write('test_regression.py', regression+'    def test_negative(self): self.assertEqual(triple(-2),-6)\n'),
            tests(), write('calc.py', 'def triple(x):\n    return x * 3\n'), tests(),
            action('update_notes', review='Positive, zero and negative regression assertions now pass.'), action('submit')],
            'expert': [action('read_files', requests=[{'path':'calc.py'}, {'path':'test_regression.py'}]),
                       completion(self.report())]}
        result = self.agent(scripts).solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        self.assertEqual([r['tests_failed'] for r in self.dispatcher.results], [1, 0])
        state = json.loads((self.home/'logs/0/working_state.json').read_text())
        self.assertEqual([t['passed'] for t in state['test_history']], [False, True, False, True])
        self.assertEqual(state['review_history'][0]['status'], 'accepted')
        self.assertIn('test_negative', (self.repo/'test_regression.py').read_text())
        solver_messages = '\n'.join(json.dumps(r['messages']) for r in self.client.requests if r['model']=='standard')
        self.assertIn('preserve negative signs', solver_messages)
        self.assertEqual(len(state['round_history']), 2)


if __name__ == '__main__':
    unittest.main()
