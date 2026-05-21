"""Unit tests for sentinel.depth_aggregation.

Run with:
    .venv\\Scripts\\python.exe test_depth_aggregation.py
"""
from __future__ import annotations

from sentinel.config import DepthConfig
from sentinel.depth_aggregation import DocumentDepth, aggregate_document_depth


def test_aggregate_sums_clause_depths() -> None:
    config = DepthConfig()
    result = aggregate_document_depth([1, 2, 3], config)
    assert isinstance(result, DocumentDepth)
    assert result.raw_sum == 6
    assert isinstance(result.raw_sum, int)
    assert result.capped == 6


def test_aggregate_empty_is_zero_zero() -> None:
    config = DepthConfig()
    result = aggregate_document_depth([], config)
    assert result.raw_sum == 0
    assert result.capped == 0


def test_aggregate_float_depths_round_to_int() -> None:
    # Mock depth model emits floats (len(words) / 3.0). They must round.
    config = DepthConfig()
    result = aggregate_document_depth([1.4, 1.6, 0.3], config)
    # round(1.4)=1, round(1.6)=2, round(0.3)=0 → sum 3
    assert result.raw_sum == 3


def test_aggregate_negative_depths_floor_at_zero() -> None:
    # Real depth model shouldn't emit negatives, but be defensive.
    config = DepthConfig()
    result = aggregate_document_depth([-3.0, 2.0, -1.0], config)
    assert result.raw_sum == 2  # only the positive contributes


def test_aggregate_cap_clamps_to_max_doc_depth() -> None:
    config = DepthConfig(max_doc_depth=5)
    result = aggregate_document_depth([3, 3, 3], config)
    assert result.raw_sum == 9
    assert result.capped == 5


def test_aggregate_below_cap_preserves_sum() -> None:
    config = DepthConfig(max_doc_depth=100)
    result = aggregate_document_depth([2, 2, 2], config)
    assert result.raw_sum == 6
    assert result.capped == 6


def test_aggregate_replaces_max_behavior() -> None:
    # The old pipeline used max(...). For [1, 5, 2]: sum=8, max=5 — different.
    config = DepthConfig()
    result = aggregate_document_depth([1, 5, 2], config)
    assert result.raw_sum == 8
    assert result.raw_sum != max([1, 5, 2])


TESTS = [
    test_aggregate_sums_clause_depths,
    test_aggregate_empty_is_zero_zero,
    test_aggregate_float_depths_round_to_int,
    test_aggregate_negative_depths_floor_at_zero,
    test_aggregate_cap_clamps_to_max_doc_depth,
    test_aggregate_below_cap_preserves_sum,
    test_aggregate_replaces_max_behavior,
]


def main() -> None:
    passed = 0
    failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print()
    print(f"{passed} passed, {failed} failed (out of {len(TESTS)})")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
