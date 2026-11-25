"""Agent settings. API credentials, URL and model IDs come from course env vars."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    solver_model: str = "standard"  # starter, standard, or expert
    reviewer_enabled: bool = True  # One independent review after failed grading only.
    reviewer_model: str = "expert"
    reviewer_output_tokens: int = 16384
    reviewer_context_chars: int = 64000
    reviewer_max_calls: int = 1  # Per failed grading; calls share task/run budgets.
    reviewer_min_remaining_s: int = 120
    reviewer_repair_reserve_usd: float = 0.05
    workers: int = 5
    api_concurrency: int = 5
    max_model_calls: int = 400
    max_evaluations: int = 5
    task_timeout_s: int = 1200
    api_timeout_s: int = 180
    api_max_retries: int = 3  # Three retries after the initial transient failure.
    starter_output_tokens: int = 8192
    standard_output_tokens: int = 16284
    expert_output_tokens: int = 16384
    run_budget_usd: float = 10.0
    task_budget_usd: float = 0.75
    tool_output_chars: int = 24000
    read_default_lines: int = 200
    read_max_lines: int = 500
    notes_entry_max_chars: int = 2000  # Maximum size of one incremental note entry.
    notes_context_chars: int = 12000  # Keep task notes from overwhelming the solver prompt.
    context_max_chars: int = 120000
    context_recent_exchanges: int = 6
    evidence_context_chars: int = 24000
    log_dir: str = "madsOpt_logs"

    def __post_init__(self):
        if self.solver_model not in ("starter", "standard", "expert"):
            raise ValueError("solver_model must be starter, standard, or expert")
        if self.reviewer_model not in ("starter", "standard", "expert"):
            raise ValueError("reviewer_model must be starter, standard, or expert")
        if type(self.reviewer_enabled) is not bool:
            raise ValueError("reviewer_enabled must be boolean")
        if self.reviewer_max_calls != 1:
            raise ValueError("This simple loop supports exactly one review call per failed grading")
        if not 0 <= self.reviewer_repair_reserve_usd < self.task_budget_usd:
            raise ValueError("reviewer_repair_reserve_usd must be nonnegative and below task budget")
        for name, value in vars(self).items():
            if type(value) is int and value < (0 if name == 'api_max_retries' else 1):
                raise ValueError(f"{name} must be positive")
        if self.workers > 5:
            raise ValueError("The dispatcher permits at most five live tasks")

    @property
    def output_tokens(self):
        return getattr(self, f"{self.solver_model}_output_tokens")


CONFIG = AgentConfig()
