"""Contract for configurable minimum context floor (model.min_context_length).

Default stays 64K; an explicit small floor flows into the compaction trigger so
8K single-purpose agents compress against their real window instead of the default.
"""

from agent.context_compressor import ContextCompressor
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    minimum_context_length_for,
    resolve_minimum_context_length,
)


class _Agent:
    def __init__(self, floor=None):
        if floor is not None:
            self._min_context_length = floor


def test_resolve_minimum_context_length_defaults_and_rejects_garbage():
    assert resolve_minimum_context_length() == MINIMUM_CONTEXT_LENGTH
    assert resolve_minimum_context_length(8192) == 8192
    assert resolve_minimum_context_length("8192") == 8192
    for bad in (None, 0, -5, True, False, "much", "", object()):
        assert resolve_minimum_context_length(bad) == MINIMUM_CONTEXT_LENGTH
    assert minimum_context_length_for() == MINIMUM_CONTEXT_LENGTH
    assert minimum_context_length_for(_Agent(8192)) == 8192
    assert minimum_context_length_for(_Agent("nope")) == MINIMUM_CONTEXT_LENGTH


def test_threshold_tokens_honors_custom_floor_not_default():
    window, max_tokens = 20_000, 1024
    default_trigger = ContextCompressor._compute_threshold_tokens(window, 0.50, max_tokens)
    custom_trigger = ContextCompressor._compute_threshold_tokens(window, 0.50, max_tokens, 4000)
    assert custom_trigger < default_trigger
    assert custom_trigger == int((window - max_tokens) * 0.50)
    compressor = ContextCompressor(
        model="t", threshold_percent=0.50, quiet_mode=True, max_tokens=max_tokens,
        minimum_context_length=4000,
    )
    compressor._resolved_context_length = window
    assert compressor.threshold_tokens == custom_trigger
