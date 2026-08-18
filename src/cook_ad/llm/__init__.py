"""LLM comparison baseline: render a trial as text, ask a model about it one step at a time.

A baseline to measure the HSMM detector against, NOT a component of it -- nothing in
`cook_ad.anomaly` imports this package. See docs/llm.md.
"""
from cook_ad.llm.client import BudgetExceeded, ChatClient, LLMError, QuotaExhausted
from cook_ad.llm.detect import Verdict

__all__ = ["BudgetExceeded", "ChatClient", "LLMError", "QuotaExhausted", "Verdict"]
