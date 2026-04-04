"""
Thread-safe, asyncio-compatible cost tracking.
Stores token counts and cost in a ContextVar so that any nested LLM call
can transparently append to the current request's cost total.
"""
import contextvars
from dataclasses import dataclass

@dataclass
class CostReport:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

# Context variable to hold cost for the current pipeline run
_cost_tracker: contextvars.ContextVar[CostReport | None] = contextvars.ContextVar(
    "cost_tracker", default=None
)

def start_tracking() -> None:
    """Initialize a new cost report for the current request context."""
    _cost_tracker.set(CostReport())

def get_report() -> CostReport | None:
    """Get the current context's cost report, if tracking is active."""
    return _cost_tracker.get()

def record_usage(model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    """Add usage to the current context's cost report."""
    report = _cost_tracker.get()
    if not report:
        return

    report.prompt_tokens += prompt_tokens
    report.completion_tokens += completion_tokens
    report.total_tokens += total_tokens

    # Approximate pricing (e.g., gpt-4o-mini is $0.150/1M input, $0.600/1M output)
    cost = 0.0
    if "mini" in model:
        cost = (prompt_tokens / 1_000_000 * 0.150) + (completion_tokens / 1_000_000 * 0.600)
    elif "gpt-4o" in model:
        cost = (prompt_tokens / 1_000_000 * 5.0) + (completion_tokens / 1_000_000 * 15.0)
    elif "gpt-3.5" in model:
        cost = (prompt_tokens / 1_000_000 * 0.5) + (completion_tokens / 1_000_000 * 1.5)
    
    report.estimated_cost += cost