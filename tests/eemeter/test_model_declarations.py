"""Every serializable opendsm model class declares its own model frequency and a model choice
no other class uses."""

import inspect
import re

import pytest

import opendsm.drmeter
import opendsm.eemeter.models
from opendsm.common.const import ModelFrequency
from opendsm.drmeter import CaltrackDRModel
from opendsm.eemeter.common.exceptions import ModelLoadError
from opendsm.eemeter.models import (
    BillingModel,
    BillingWeightedModel,
    DailyModel,
    HourlyCaltrackModel,
    HourlyModel,
)



EXPECTED_DECLARATIONS = {
    DailyModel: (ModelFrequency.DAILY, "daily"),
    BillingModel: (ModelFrequency.BILLING, "billing"),
    BillingWeightedModel: (ModelFrequency.BILLING, "billing_weighted"),
    HourlyModel: (ModelFrequency.HOURLY, "hourly"),
    HourlyCaltrackModel: (ModelFrequency.HOURLY, "hourly_caltrack"),
    CaltrackDRModel: (ModelFrequency.HOURLY, "drmeter_caltrack"),
}


def _qualified(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _serializable_model_classes():
    """The exported classes that serialize (define both to_dict and from_dict), plus every
    opendsm subclass of them."""
    found = {
        obj
        for module in (opendsm.eemeter.models, opendsm.drmeter)
        for obj in vars(module).values()
        if inspect.isclass(obj) and hasattr(obj, "to_dict") and hasattr(obj, "from_dict")
    }
    pending = list(found)
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub.__module__.startswith("opendsm.") and sub not in found:
                found.add(sub)
                pending.append(sub)

    return found


def test_every_serializable_model_class_is_in_the_table():
    found = {_qualified(cls) for cls in _serializable_model_classes()}
    expected = {_qualified(cls) for cls in EXPECTED_DECLARATIONS}

    assert found == expected, (
        f"missing from the table: {sorted(found - expected)}; "
        f"no longer serializable: {sorted(expected - found)}"
    )


@pytest.mark.parametrize("model_cls", list(EXPECTED_DECLARATIONS), ids=_qualified)
def test_model_class_declares_its_own_model_frequency(model_cls):
    declared = vars(model_cls).get("model_frequency")
    expected, _ = EXPECTED_DECLARATIONS[model_cls]

    assert declared == expected, (
        f"{_qualified(model_cls)} declares model_frequency={declared!r} in its own body, "
        f"expected {expected!r}"
    )


@pytest.mark.parametrize("model_cls", list(EXPECTED_DECLARATIONS), ids=_qualified)
def test_model_class_declares_its_own_model_choice(model_cls):
    declared = vars(model_cls).get("model_choice")
    _, expected = EXPECTED_DECLARATIONS[model_cls]

    assert declared == expected, (
        f"{_qualified(model_cls)} declares model_choice={declared!r} in its own body, "
        f"expected {expected!r}"
    )


def test_model_choices_are_unique():
    choices = [vars(cls).get("model_choice") for cls in _serializable_model_classes()]
    duplicates = sorted({choice for choice in choices if choices.count(choice) > 1})

    assert not duplicates, f"model_choice values used by more than one class: {duplicates}"


FOREIGN_PAYLOADS = [
    (model_cls, written_by)
    for model_cls, (_, choice) in EXPECTED_DECLARATIONS.items()
    for _, written_by in [*EXPECTED_DECLARATIONS.values(), (None, "foreign")]
    if written_by != choice
]


@pytest.mark.parametrize(
    "model_cls, written_by",
    FOREIGN_PAYLOADS,
    ids=[f"{_qualified(cls)}-{written_by}" for cls, written_by in FOREIGN_PAYLOADS],
)
def test_from_dict_rejects_a_payload_written_by_another_model(model_cls, written_by):
    """A payload whose `model_choice` names any other model, declared or not, raises before any
    field is read."""
    _, choice = EXPECTED_DECLARATIONS[model_cls]
    match = re.escape(f"'{written_by}' model; this class loads '{choice}'")

    with pytest.raises(ModelLoadError, match=match):
        model_cls.from_dict({"model_choice": written_by})
