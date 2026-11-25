"""Adversarial regressions for failures observed in real agent traces."""

import json
import unittest
from pathlib import Path

import test_workflow as fixture
from src.regressions import describe_test, differences, source_fingerprint
from src.tools import is_test_command, notes_tool


class RegressionTests(unittest.TestCase):
    setUp = fixture.WorkflowTests.setUp
    agent = fixture.WorkflowTests.agent
    ready = fixture.WorkflowTests.ready
    smoke_arguments = staticmethod(fixture.WorkflowTests.smoke_arguments)

    def perform(self, name='run_tests', **arguments):
        agent = self.agent()
        message = fixture.action(name, **arguments)['choices'][0]['message']
        results, _ = agent.execute_tools(self.state, self.tools, message, {name})
        return json.loads(results[0]['content'])

    def regression(self, count=20):
        text = 'import unittest\nfrom calc import triple\nclass Regression(unittest.TestCase):\n'
        text += ''.join(f'    def test_{i}(self): self.assertEqual(triple({i}), {i} * 3)\n' for i in range(count))
        return text

    def test_twenty_to_eight_is_caught_before_any_grading(self):
        path = self.repo / 'test_regression.py'
        path.write_text(self.regression(20))
        self.assertTrue(self.tools.regressions.capture()['ok'])
        path.write_text(self.regression(8))
        report = self.tools.regressions.capture()
        self.assertFalse(report['ok'])
        self.assertEqual(len(report['changes_requiring_review'][0]['changes']), 12)
        self.assertEqual(self.state.grades, [])
        self.assertEqual(len(list((self.state.directory/'regression_snapshots').glob('*.txt'))), 2)
        self.state.save()
        manifest = json.loads((self.state.directory/'regression_manifest.json').read_text())
        self.assertEqual(len(manifest['required']['test_regression.py']['cases']), 20)

    def test_additive_cases_preserved_and_deleted_file_cannot_be_waived(self):
        path = self.repo / 'test_regression.py'
        path.write_text(self.regression(2))
        self.tools.regressions.capture()
        path.write_text(self.regression(3))
        self.assertTrue(self.tools.regressions.capture()['ok'])
        self.assertEqual(len(self.state.regression_required['test_regression.py']['cases']), 3)
        path.unlink()
        report = self.tools.regressions.capture()
        warning = report['changes_requiring_review'][0]
        with self.assertRaises(ValueError):
            self.tools.regressions.accept_corrections([{**warning, 'requirement': 'x', 'reason': 'x'}])
        path.write_text(self.regression(3))
        self.assertTrue(self.tools.regressions.capture()['ok'])

    def test_assertion_skip_and_parameter_changes_are_not_silent(self):
        original = 'import pytest\n@pytest.mark.parametrize("x", [1,2])\ndef test_value(x):\n    assert x > 0\n'
        for changed in (original.replace('> 0', '>= 0'),
                        original.replace('def test_value', '@pytest.mark.skip\ndef test_value'),
                        original.replace('[1,2]', '[1]'),
                        original.replace('    assert x > 0', '    pass')):
            with self.subTest(changed=changed):
                self.assertTrue(differences(describe_test('test_a.py', original), describe_test('test_a.py', changed)))
        self.assertFalse(differences(describe_test('test_a.py', original),
            describe_test('test_a.py', original + '\ndef test_more():\n    assert True\n')))

    def test_oracle_correction_requires_current_hashes_and_requirement_reason(self):
        path = self.repo / 'test_regression.py'
        path.write_text(self.regression(2))
        self.tools.regressions.capture()
        path.write_text(self.regression(2).replace('0 * 3', '0'))
        warning = self.tools.regressions.capture()['changes_requiring_review'][0]
        for changes in ({'before_sha256': 'stale'}, {'reason': ''}, {'requirement': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.tools.regressions.accept_corrections([{**warning, 'requirement': 'triple(0) is 0',
                    'reason': 'Equivalent explicit expectation', **changes}])
        result = self.tools.regressions.accept_corrections([{**warning,
            'requirement': 'The task requires multiplication by three.',
            'reason': 'For input zero, the equivalent expected value is explicitly zero.'}])
        self.assertTrue(result['ok'])
        self.assertEqual(len(self.state.test_corrections), 1)

    def test_changed_existing_test_is_compared_with_git_baseline(self):
        import subprocess
        path = self.repo/'test_existing.py'
        path.write_text('def test_old():\n    assert 2 == 2\n')
        subprocess.run(['git', 'add', '.'], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'test baseline'], cwd=self.repo, check=True, capture_output=True)
        path.write_text('def test_new():\n    assert True\n')
        report = self.tools.regressions.capture()
        self.assertFalse(report['ok'])
        self.assertIn('test_old', str(report))

    def test_plain_unittest_cd_and_pipeline_are_recorded_without_labels(self):
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        (self.repo/'test_regression.py').write_text(self.regression(2))
        result = self.perform('bash', command='cd . && python3 -B -m unittest test_regression -v 2>&1 | tail -20')
        self.assertEqual(result['exit_code'], 0, result)
        self.assertTrue(self.state.last_local_test['passed'])
        self.assertIn('calc.py', self.state.last_local_test['covers'])
        self.assertIn('submit', result['next_action'])
        self.assertTrue(self.state.readiness_checks[-1]['passed'])
        self.assertEqual(self.agent().grade(self.state, self.tools, self.state.patch_text)['grading']['tests_failed'], 0)

    def test_test_classifier_does_not_match_strings_in_edit_commands(self):
        for command in ('pytest -q', 'cd /app && env MODE=test python3 -B -m pytest tests | tail -30',
                        'go test ./...', 'npm run test:unit -- -w 1'):
            self.assertTrue(is_test_command(command), command)
        for command in ('echo "pytest test_a.py"', 'python3 -c "print(\'pytest\')"',
                        "python3 - <<'PY'\n# pytest\nprint('edit')\nPY", 'git grep "go test"'):
            self.assertFalse(is_test_command(command), command)

    def test_failed_plain_tests_are_not_hidden_by_pipe_or_ignored_status(self):
        (self.repo/'test_regression.py').write_text(self.regression(2))
        result = self.perform('bash', command='python3 -B -m unittest test_regression -v 2>&1 | tail -20')
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse(self.state.last_local_test['passed'])
        result = self.perform(command='python3 -B -m unittest test_regression || true')
        self.assertIn('masks', result['error'])
        self.assertFalse(self.state.last_local_test['passed'])

    def test_no_tests_is_not_passing_validation(self):
        result = self.perform(command='python3 -B -m unittest discover -p nonexistent_*.py')
        self.assertIn('No tests executed', result['error'])
        self.assertFalse(self.state.last_local_test['passed'])

    def test_all_skipped_suite_is_not_passing_validation(self):
        (self.repo/'test_skip.py').write_text('import unittest\n@unittest.skip("not exercised")\n'
            'class Example(unittest.TestCase):\n    def test_value(self): self.assertTrue(False)\n')
        result = self.perform(command='python3 -B -m unittest test_skip')
        self.assertIn('All selected tests were skipped', result['error'])
        self.assertFalse(self.state.last_local_test['passed'])

    def test_non_python_test_removal_and_changed_fixture_are_flagged(self):
        before = describe_test('x_test.go', 'func TestOld(t *testing.T) {\n    t.Fatal("regression")\n}\n')
        after = describe_test('x_test.go', 'func TestNew(t *testing.T) {\n}\n')
        self.assertTrue(differences(before, after))
        original = 'import unittest\nVALUE=3\ndef test_value():\n    assert VALUE == 3\n'
        self.assertTrue(differences(describe_test('test_value.py', original),
            describe_test('test_value.py', original.replace('VALUE=3', 'VALUE=2'))))

    def test_premature_repair_does_not_discard_valid_initial_plan(self):
        self.assertNotIn('repair_review', notes_tool(self.state)['function']['parameters']['properties'])
        self.state.update_notes({'findings': ['calc.py:2 wrong multiplier'], 'plan': ['Fix triple'],
                                 'repair_review': fixture.repair_review()})
        self.assertEqual(self.state.notes['plan'], ['Fix triple'])
        self.assertIsNone(self.state.repair_review)
        self.state.record_grade('patch', {'tests_failed': 1, 'tests_total': 8, 'attempt': 0})
        self.assertIn('repair_review', notes_tool(self.state)['function']['parameters']['properties'])

    def test_expert_counterexample_already_passing_is_not_a_reproduced_defect(self):
        agent = self.agent()
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        patch = agent.current_patch(self.state)
        self.state.review_history = [{'round': 1, 'status': 'accepted',
            'source_sha256': source_fingerprint(patch)}]
        (self.repo/'test_regression.py').write_text(self.regression(2))
        result = self.perform(command='python3 -B -m unittest test_regression -v', review_case=True)
        self.assertTrue(result['review_probe']['on_reviewed_source'])
        self.assertEqual(result['review_probe']['outcome'], 'already_passes')
        self.assertIn('do NOT apply', result['next_action'])

    def test_genuine_counterexample_and_setup_error_are_distinguished(self):
        agent = self.agent()
        self.state.review_history = [{'round': 1, 'status': 'accepted',
            'source_sha256': source_fingerprint(agent.current_patch(self.state))}]
        (self.repo/'test_regression.py').write_text(self.regression(2))
        result = self.perform(command='python3 -B -m unittest test_regression -v', review_case=True)
        self.assertEqual(result['review_probe']['outcome'], 'reproduced')
        self.assertTrue(result['review_probe']['on_reviewed_source'])
        result = self.perform(command='python3 -m unittest missing_test_module', review_case=True)
        self.assertEqual(result['review_probe']['outcome'], 'inconclusive')

    def test_submit_blocks_lost_tests_even_when_smaller_suite_passes(self):
        agent = self.agent()
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        path = self.repo/'test_regression.py'
        path.write_text(self.regression(20))
        self.tools.regressions.capture()
        path.write_text(self.regression(8))
        self.perform(command='python3 -B -m unittest test_regression')
        feedback = agent.grade(self.state, self.tools, self.state.patch_text)
        self.assertIn('Restore preserved regressions', feedback['error'])
        self.assertEqual(self.dispatcher.results, [])

    def test_review_sources_retrieve_complete_entry_point_and_distant_helper(self):
        import os
        from src.reviews import locate_review_symbols
        text = ('def triple(x):\n    return choose(x)\n' + '\n' * 250 +
                'def choose(x):\n    return x * 3\n')
        (self.repo/'calc.py').write_text(text)
        previous = os.getcwd()
        try:
            os.chdir(self.repo)
            locations = locate_review_symbols(['calc.py'], ['triple'], '')
        finally:
            os.chdir(previous)
        self.assertEqual([r['start_line'] for r in locations], [1, 253])
        self.assertEqual([r['num_lines'] for r in locations], [2, 2])
        packet = self.agent().review_packet(self.state, 'fresh source', 'diff', 'failed grading')
        self.assertIn('Preserved regression inventory', packet)

    def test_repair_with_same_green_suite_is_rejected_until_new_case_added(self):
        agent = self.agent()
        self.dispatcher.grade_script = [1, 0]
        path = self.repo/'test_regression.py'
        (self.repo/'calc.py').write_text('def triple(x):\n    return abs(x) * 3\n')
        path.write_text(self.regression(2))
        self.perform(command='python3 -B -m unittest test_regression')
        self.assertIn('grading', agent.grade(self.state, self.tools, self.state.patch_text))
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        self.perform(command='python3 -B -m unittest test_regression')
        feedback = agent.grade(self.state, self.tools, self.state.patch_text)
        self.assertIn('new discriminating regression', feedback['error'])
        path.write_text(self.regression(2) + '    def test_negative(self): self.assertEqual(triple(-2), -6)\n')
        self.perform(command='python3 -B -m unittest test_regression')
        # Local mode: evidence succeeds without repair_notes or prose approval.
        self.assertIsNone(self.state.repair_review)
        feedback = agent.grade(self.state, self.tools, self.state.patch_text)
        self.assertEqual(feedback['grading']['tests_failed'], 0)

    def test_case_after_source_edit_is_not_claimed_as_original_reproduction(self):
        agent = self.agent()
        self.state.review_history = [{'round': 1, 'status': 'accepted',
            'source_sha256': source_fingerprint(agent.current_patch(self.state))}]
        (self.repo/'calc.py').write_text('def triple(x):\n    return x * 3\n')
        (self.repo/'test_regression.py').write_text(self.regression(2))
        result = self.perform(command='python3 -B -m unittest test_regression', review_case=True)
        self.assertFalse(result['review_probe']['on_reviewed_source'])
        self.assertEqual(result['review_probe']['outcome'], 'inconclusive')

    def test_end_to_end_new_tools_and_actionable_expert_counterexample(self):
        import shlex
        regression = self.regression(2)
        def write(path, text):
            return fixture.action('bash', command='python3 -c ' + shlex.quote(
                f'from pathlib import Path; Path({path!r}).write_text({text!r})'))
        def tests(probe=False):
            return fixture.action('run_tests', command='python3 -B -m unittest test_regression -v', review_case=probe)
        self.dispatcher.grade_script = [1, 0]
        scripts = {'standard': [
            fixture.action('update_notes', findings=['calc.py uses wrong multiplier'],
                           plan=['Write regression and fix triple'], repair_review=fixture.repair_review()),
            write('test_regression.py', regression), tests(),
            write('calc.py', 'def triple(x):\n    return abs(x) * 3\n'), tests(), fixture.action('submit'),
            write('test_regression.py', regression +
                  '    def test_negative(self): self.assertEqual(triple(-2), -6)\n'), tests(True),
            write('calc.py', 'def triple(x):\n    return x * 3\n'), tests(), fixture.action('submit')],
            'expert': [fixture.completion(fixture.WorkflowTests.report())]}
        result = self.agent(scripts).solve(self.task)
        self.assertEqual(result['tests_failed'], 0)
        state = json.loads((self.home/'logs/0/working_state.json').read_text())
        self.assertEqual([r['tests_failed'] for r in state['grades']], [1, 0])
        self.assertEqual(state['review_probe']['outcome'], 'reproduced')
        self.assertTrue(state['review_probe']['on_reviewed_source'])
        self.assertIsNone(state['repair_review'])
        self.assertEqual(len(state['regression_inventory']['test_regression.py']['case_ids']), 3)
        self.assertEqual(len([r for r in self.client.requests if r['model'] == 'expert']), 1)


if __name__ == '__main__':
    unittest.main()
