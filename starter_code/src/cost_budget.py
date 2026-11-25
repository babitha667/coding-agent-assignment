"""Buffered token estimates and thread-safe admission for course API requests.

Unknown/failed requests keep their full reservation: a timeout is not evidence
that the server did no billable work. SDK retries must be disabled by callers.
"""

import json
import math
import threading
from dataclasses import dataclass


# Dollars per million tokens: input, cached input, output (course assignment).
RATES = {"starter": (0.01, 0.0005, 0.04),
         "standard": (0.08, 0.004, 0.28),
         "expert": (2.00, 0.10, 10.00)}


class BudgetExceeded(RuntimeError):
    pass


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def input_upper_bound(messages, options):
    """UTF-8 bytes + framing allowance, deliberately not chars/4.

    This is conservative for ordinary byte-based tokenizers, not an assertion
    about the undisclosed course tokenizer. Log actual usage to detect overruns.
    """
    payload = {"messages": messages, **options}
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")) + 1024 + 64 * len(messages)


def estimated_input_tokens(messages, options, density=None):
    """Estimate unknown-tokenizer input, with 25% headroom plus framing.

    ASCII source/prose starts at three bytes per token; non-ASCII bytes are
    reserved individually. Once usage is known, use the highest observed token
    density for this model. This is an estimate, NOT a guaranteed upper bound.
    Transport JSON bytes are not themselves model tokens.
    """
    payload = json.dumps({"messages": messages, **options}, ensure_ascii=False, default=str).encode("utf-8")
    ascii_bytes = sum(byte < 128 for byte in payload)
    nonascii_bytes = len(payload) - ascii_bytes
    tokens = (ascii_bytes / 3 + nonascii_bytes) if density is None else (
        ascii_bytes * density + nonascii_bytes * max(1.0, density))
    return max(1, math.ceil(tokens * 1.25) + 1024 + 64 * len(messages))


def input_sample(messages, options, usage):
    prompt = field(usage, "prompt_tokens")
    if type(prompt) is not int or prompt < 0:
        return None
    return {"messages": json.dumps(messages, ensure_ascii=False, default=str),
            "options": json.dumps(options, sort_keys=True, ensure_ascii=False, default=str),
            "count": len(messages), "prompt_tokens": prompt}


def measured_input_bound(messages, options, sample=None, density=None):
    """Reuse measured tokens for an identical prefix; estimate other input.

    Changed roots, compaction or a new review use buffered token estimates,
    not one-token-per-transport-byte reservations. No future cache hit is assumed.
    """
    fallback = estimated_input_tokens(messages, options, density)
    if sample is None or len(messages) < sample["count"]:
        return fallback
    settings = json.dumps(options, sort_keys=True, ensure_ascii=False, default=str)
    prefix = json.dumps(messages[:sample["count"]], ensure_ascii=False, default=str)
    if prefix != sample["messages"]:
        return fallback
    control_bytes = 0
    if settings != sample["options"]:
        prior_options = json.loads(sample["options"])
        prior_body = {k: v for k, v in prior_options.items() if k != "tool_choice"}
        current_body = {k: v for k, v in options.items() if k != "tool_choice"}
        if prior_body != current_body:
            return fallback
        # Reserve the full serialized control change, without assuming a cache hit.
        control_bytes = len(json.dumps({"before": prior_options.get("tool_choice"),
            "after": options.get("tool_choice")}, ensure_ascii=False).encode("utf-8"))
    added = messages[sample["count"]:]
    increment = estimated_input_tokens(added, {}, density)
    return sample["prompt_tokens"] + increment + control_bytes


def token_cost(tier, prompt, completion, cached=0):
    inp, cache, out = RATES[tier]
    return ((prompt - cached) * inp + cached * cache + completion * out) / 1_000_000


@dataclass
class Reservation:
    task: int
    tier: str
    role: str
    input_bound: int
    output_limit: int
    amount: float
    settled: bool = False
    model: str = ""
    input_bytes: int = 0


class CostBudget:
    def __init__(self, run_limit=10.0, task_limit=0.50, expert_limit=0.18,
                 expert_calls=3, task_calls=400):
        for value in (run_limit, task_limit, expert_limit):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Budget limits must be finite and positive")
        if run_limit > 10.0 or task_limit > run_limit or expert_limit > task_limit:
            raise ValueError("Keep run <= $10, task <= run, and Expert <= task")
        if type(expert_calls) is not int or expert_calls < 0:
            raise ValueError("Expert-call allowance must be a nonnegative integer")
        if type(task_calls) is not int or task_calls < 1:
            raise ValueError("task_calls must be a positive integer")
        self.run_limit, self.task_limit = run_limit, task_limit
        self.expert_limit, self.expert_calls = expert_limit, expert_calls
        self.task_calls = task_calls
        self.lock = threading.Lock()
        self.spent = self.reserved = 0.0
        self.tasks = {}
        self.input_densities = {}  # Keyed by (task, actual model ID), under the same lock.

    def estimate_input(self, task, messages, options, sample=None):
        with self.lock:
            density = self.input_densities.get((task, options.get("model")))
        return measured_input_bound(messages, options, sample, density)

    def _task(self, task):
        return self.tasks.setdefault(task, dict(spent=0.0, reserved=0.0,
            expert_spent=0.0, expert_reserved=0.0, calls=0, expert_calls=0,
            estimated_billed=0.0, uncached_spent=0.0, unknown_requests=0, usage_overruns=0))

    def reserve(self, task, tier, role, messages, options, output_limit, input_bound=None,
                repair_reserve=0.0):
        if tier not in RATES or type(output_limit) is not int or output_limit < 1:
            raise ValueError("Unknown model tier or invalid output limit")
        if not math.isfinite(repair_reserve) or repair_reserve < 0:
            raise ValueError("Repair reserve must be finite and nonnegative")
        upper = self.estimate_input(task, messages, options) if input_bound is None else input_bound
        if type(upper) is not int or upper < 1:
            raise ValueError("Input token reservation must be a positive integer")
        with self.lock:
            state = self._task(task)
            if state["calls"] >= self.task_calls:
                raise BudgetExceeded("task model-call limit reached")
            available = min(self.run_limit - self.spent - self.reserved,
                            self.task_limit - state["spent"] - state["reserved"])
            if tier == "expert":
                if state["expert_calls"] >= self.expert_calls:
                    raise BudgetExceeded("task Expert-call limit reached")
                available = min(available, self.expert_limit - state["expert_spent"] - state["expert_reserved"])
            # Recheck under the reservation lock; another worker may have used
            # shared run funds since a caller's earlier affordability estimate.
            available -= repair_reserve
            input_cost = token_cost(tier, upper, 0)
            affordable = math.floor((available - input_cost) * 1_000_000 / RATES[tier][2])
            allowed = min(output_limit, affordable)
            if allowed < min(output_limit, 512):
                raise BudgetExceeded(f"insufficient {tier} request budget (including input and in-flight requests)")
            amount = token_cost(tier, upper, allowed)
            ticket = Reservation(task, tier, role, upper, allowed, amount)
            ticket.model = options.get("model", tier)
            ticket.input_bytes = len(json.dumps({"messages": messages, **options},
                ensure_ascii=False, default=str).encode("utf-8"))
            self.reserved += amount
            state["reserved"] += amount
            state["calls"] += 1
            if tier == "expert":
                state["expert_reserved"] += amount
                state["expert_calls"] += 1
            return ticket

    def settle(self, ticket, usage=None):
        prompt, output = field(usage, "prompt_tokens"), field(usage, "completion_tokens")
        known = all(type(x) is int and x >= 0 for x in (prompt, output))
        cached = field(field(usage, "prompt_tokens_details"), "cached_tokens")
        if cached is None:
            # The course also exposes these aliases when SDK nested details are absent.
            cached = field(usage, "cached_tokens", field(usage, "cache_read_tokens"))
        cache_known = known and type(cached) is int and 0 <= cached <= prompt
        uncached_cost = token_cost(ticket.tier, prompt, output) if known else ticket.amount
        billed_estimate = token_cost(ticket.tier, prompt, output, cached if cache_known else 0) if known else ticket.amount
        control_cost = billed_estimate  # Refund only savings reported by this completed request.
        with self.lock:
            if ticket.settled:
                raise RuntimeError("API reservation already settled")
            ticket.settled = True
            state = self._task(ticket.task)
            if known and prompt > 0 and ticket.input_bytes > 0:
                density = prompt / ticket.input_bytes
                key = (ticket.task, ticket.model)
                self.input_densities[key] = max(density, self.input_densities.get(key, 0))
            self.reserved = max(0.0, self.reserved - ticket.amount)
            state["reserved"] = max(0.0, state["reserved"] - ticket.amount)
            self.spent += control_cost
            state["spent"] += control_cost
            state["estimated_billed"] += billed_estimate
            state["uncached_spent"] += uncached_cost
            state["unknown_requests"] += not known
            state["usage_overruns"] += bool(known and uncached_cost > ticket.amount + 1e-9)
            if ticket.tier == "expert":
                state["expert_reserved"] = max(0.0, state["expert_reserved"] - ticket.amount)
                state["expert_spent"] += control_cost
            return {"role": ticket.role, "tier": ticket.tier,
                "reservation_usd": ticket.amount, "control_cost_usd": control_cost,
                "billed_estimate_usd": billed_estimate, "usage_known": known,
                "uncached_cost_usd": uncached_cost,
                "cached_tokens_known": cache_known, "input_tokens_bound": ticket.input_bound,
                "output_token_limit": ticket.output_limit,
                "usage_exceeded_reservation": known and uncached_cost > ticket.amount + 1e-9,
                "input_estimate_exceeded": known and prompt > ticket.input_bound,
                "task": dict(state), "run_control_cost_usd": self.spent,
                "run_reserved_usd": self.reserved}

    def snapshot(self, task):
        with self.lock:
            state = dict(self._task(task))
            return {**state, "task_limit_usd": self.task_limit,
                "task_remaining_usd": max(0.0, self.task_limit - state["spent"] - state["reserved"]),
                "expert_calls_remaining": max(0, self.expert_calls - state["expert_calls"]),
                "run_limit_usd": self.run_limit, "run_control_cost_usd": self.spent,
                "run_remaining_usd": max(0.0, self.run_limit - self.spent - self.reserved)}
