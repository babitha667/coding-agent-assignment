"""Focused tests for compacting captured repository/tool evidence."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

STARTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STARTER))

from src.agentic_loop import Agent


class EvidenceContextTests(unittest.TestCase):
    def test_context_preserves_paths_search_hits_reads_and_edits(self):
        observations = [
            {"call": 1, "tool": "list_files", "arguments": {},
             "result": {"files": ["src/target.py"]}},
            {"call": 2, "tool": "search_files", "arguments": {"pattern": "target"},
             "result": {"matches": [{"path": "src/target.py", "line": 7,
                                       "text": "def target(value):"}]}},
            {"call": 3, "tool": "read_files", "arguments": {},
             "result": {"files": [{"path": "src/target.py", "content": "7: def target(value):\n8:     return value",
                                   "total_lines": 8}]}},
            {"call": 4, "tool": "replace_text", "arguments": {},
             "result": {"edited_files": ["src/target.py"]}},
        ]

        context = json.loads(Agent.solver_evidence_context(
            observations, changed_paths=["src/target.py"], max_chars=4000))

        self.assertEqual(context["repository_paths"], ["src/target.py"])
        self.assertEqual(context["searches"][0]["matches"][0]["line"], 7)
        self.assertEqual(context["read_snippets"][0]["start_line"], 7)
        self.assertEqual(context["edited_files"][0]["edited_files"], ["src/target.py"])
        self.assertIn("no-match is not proof", context["warning"])

    def test_context_is_bounded_and_reports_omissions(self):
        observations = [{"call": i, "tool": "read_files", "arguments": {},
                         "result": {"files": [{"path": f"src/{i}.py",
                                               "content": f"1: {'x' * 1500}",
                                               "total_lines": 1}]}}
                        for i in range(20)]

        encoded = Agent.solver_evidence_context(observations, max_chars=2500)
        context = json.loads(encoded)

        self.assertLessEqual(len(encoded), 2500)
        self.assertGreater(context["omitted_evidence"]["reads"], 0)

    def test_compaction_injects_evidence_only_when_needed(self):
        agent = SimpleNamespace(config=SimpleNamespace(
            context_max_chars=500, context_recent_exchanges=1, notes_max_chars=100))
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old " * 100},
            {"role": "tool", "tool_call_id": "x", "content": "result " * 100},
        ]

        compacted = Agent.compact(agent, messages, [], [], '{"read_snippets": []}')

        self.assertIn("Captured repository evidence", compacted[2]["content"])
        self.assertEqual(compacted[0]["content"], "system")
        self.assertEqual(compacted[1]["content"], "task")


if __name__ == "__main__":
    unittest.main()
