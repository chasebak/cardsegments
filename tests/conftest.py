"""Shared fixtures.

buylist_recompute: one in-memory run of `python -m cardsegments.buylist
build` (two logistic rungs x 10 CV seeds x 2 signals, about 3 s), shared
by tests/test_buylist.py and tests/test_explorer.py so CI pays for it once.
It returns the payload exactly as build would write it (before rounding)
and the per-card probabilities, which stay in memory and never reach disk.
"""

import pytest

from cardsegments import buylist


@pytest.fixture(scope="session")
def buylist_recompute():
    probs = {}
    payload = buylist.run(verbose=False, probs_out=probs)
    return payload, probs
