"""Tests for the tolerant JSON snapshot comparison in ``syrupy_extensions``."""

import json

import pytest

from syrupy_extensions import (
    FLOAT_ATOL,
    TolerantJSONSnapshotExtension,
)



def _matches(extension_class, actual, expected) -> bool:
    # matches() reads only class attributes, so the syrupy-managed constructor is skipped
    extension = extension_class.__new__(extension_class)
    serialized = json.dumps(actual)
    snapshot = json.dumps(expected)
    result = extension.matches(serialized_data=serialized, snapshot_data=snapshot)

    return result


_TOLERANT = TolerantJSONSnapshotExtension.with_tolerance(
    atol=1.0, rtol=0.0, atol_keys=frozenset({"loose"}), leaf_atol={"std": 0.6}
)
_BASE = TolerantJSONSnapshotExtension


def test_base_extension_accepts_a_float_within_float_atol():
    """A float differing by less than FLOAT_ATOL matches."""
    assert _matches(_BASE, {"x": 1.0 + FLOAT_ATOL / 2}, {"x": 1.0})


def test_base_extension_rejects_a_float_beyond_tolerance():
    """A float differing by far more than FLOAT_ATOL and rtol does not match."""
    assert not _matches(_BASE, {"x": 1.01}, {"x": 1.0})


def test_ints_compare_exactly():
    """Two ints differing by 1 do not match, even under a loose atol."""
    assert not _matches(_TOLERANT, {"loose": 5}, {"loose": 4})


def test_int_equals_itself():
    """Equal ints match."""
    assert _matches(_TOLERANT, {"loose": 5}, {"loose": 5})


def test_key_missing_from_snapshot_is_a_difference():
    """A key only in the received data does not match."""
    assert not _matches(_BASE, {"a": 1.0, "b": 2.0}, {"a": 1.0})


def test_key_missing_from_received_is_a_difference():
    """A key only in the snapshot does not match."""
    assert not _matches(_BASE, {"a": 1.0}, {"a": 1.0, "b": 2.0})


def test_list_length_mismatch_is_a_difference():
    """A list index present on one side only does not match."""
    assert not _matches(_BASE, {"a": [1.0, 2.0]}, {"a": [1.0]})


def test_nan_equals_nan():
    """NaN on both sides matches."""
    assert _matches(_BASE, {"a": float("nan")}, {"a": float("nan")})


def test_nan_does_not_equal_a_number():
    """NaN against a number does not match."""
    assert not _matches(_BASE, {"a": float("nan")}, {"a": 1.0})


def test_difference_inside_nested_dict_is_found():
    """A difference below the top level is not skipped."""
    assert not _matches(_BASE, {"a": {"b": {"c": 2.0}}}, {"a": {"b": {"c": 1.0}}})


def test_non_numeric_leaves_compare_with_equality():
    """Differing strings do not match."""
    assert not _matches(_BASE, {"a": "x"}, {"a": "y"})


def test_atol_keys_use_the_wide_atol_and_other_keys_use_float_atol():
    """A 0.5 difference passes under an atol_keys key and fails under any other key."""
    assert _matches(_TOLERANT, {"loose": 1.5}, {"loose": 1.0})
    assert not _matches(_TOLERANT, {"tight": 1.5}, {"tight": 1.0})


def test_atol_keys_apply_to_list_elements_below_the_key():
    """The top-level key's atol covers every list element beneath it."""
    assert _matches(_TOLERANT, {"loose": [1.5, 2.5]}, {"loose": [1.0, 2.0]})


def test_leaf_atol_overrides_a_tighter_atol():
    """A std leaf under a FLOAT_ATOL key accepts 0.5 and rejects 0.7."""
    assert _matches(_TOLERANT, {"tight": {"std": 1.5}}, {"tight": {"std": 1.0}})
    assert not _matches(_TOLERANT, {"tight": {"std": 1.7}}, {"tight": {"std": 1.0}})


def test_leaf_atol_overrides_a_wider_atol():
    """A std leaf under a wide-atol key rejects 0.7 although the key's atol is 1.0."""
    assert not _matches(_TOLERANT, {"loose": {"std": 1.7}}, {"loose": {"std": 1.0}})


def test_leaf_atol_applies_to_list_elements_of_the_leaf_key():
    """Elements of a list stored under a leaf_atol key use that key's atol."""
    assert not _matches(_TOLERANT, {"tight": {"std": [1.7]}}, {"tight": {"std": [1.0]}})
    assert _matches(_TOLERANT, {"tight": {"std": [1.5]}}, {"tight": {"std": [1.0]}})


def test_diff_lines_reports_missing_and_non_numeric_leaves():
    """The summary names a missing key and a non-numeric mismatch without raising."""
    extension = _BASE.__new__(_BASE)
    actual = json.dumps({"a": "x", "b": 1.0})
    expected = json.dumps({"a": "y"})

    lines = "\n".join(extension.diff_lines(actual, expected))

    assert "2 value(s) outside tolerance" in lines
    assert "b: received 1.0, snapshot <missing>" in lines


@pytest.mark.parametrize("bad", ["not json", None])
def test_unparseable_data_does_not_match(bad):
    """Data that is not JSON does not match."""
    extension = _BASE.__new__(_BASE)

    assert not extension.matches(serialized_data=bad, snapshot_data="{}")
