"""Every serializable opendsm model class declares its own model frequency and a model choice
no other class uses."""

import inspect

import pytest

import opendsm.drmeter
import opendsm.eemeter.models
from opendsm.common.const import ModelFrequency
from opendsm.drmeter import CaltrackDRModel
from opendsm.eemeter.models import (
    BillingModel,
    BillingWeightedModel,
    DailyModel,
    HourlyCaltrackModel,
    HourlyModel,
)



EXPECTED_MODEL_FREQUENCY = {
    DailyModel: ModelFrequency.DAILY,
    BillingModel: ModelFrequency.BILLING,
    BillingWeightedModel: ModelFrequency.BILLING,
    HourlyModel: ModelFrequency.HOURLY,
    HourlyCaltrackModel: ModelFrequency.HOURLY,
    CaltrackDRModel: ModelFrequency.HOURLY,
}

EXPECTED_MODEL_CHOICE = {
    DailyModel: "daily",
    BillingModel: "billing",
    BillingWeightedModel: "billing_weighted",
    HourlyModel: "hourly",
    HourlyCaltrackModel: "hourly_caltrack",
    CaltrackDRModel: "drmeter_caltrack",
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
    expected = {_qualified(cls) for cls in EXPECTED_MODEL_FREQUENCY}
    choice_table = {_qualified(cls) for cls in EXPECTED_MODEL_CHOICE}

    assert choice_table == expected, (
        "the model_choice and model_frequency tables list different classes"
    )

    assert found == expected, (
        f"missing from the table: {sorted(found - expected)}; "
        f"no longer serializable: {sorted(expected - found)}"
    )


@pytest.mark.parametrize("model_cls", list(EXPECTED_MODEL_FREQUENCY), ids=_qualified)
def test_model_class_declares_its_own_model_frequency(model_cls):
    declared = vars(model_cls).get("model_frequency")

    assert declared == EXPECTED_MODEL_FREQUENCY[model_cls], (
        f"{_qualified(model_cls)} declares model_frequency={declared!r} in its own body, "
        f"expected {EXPECTED_MODEL_FREQUENCY[model_cls]!r}"
    )


@pytest.mark.parametrize("model_cls", list(EXPECTED_MODEL_CHOICE), ids=_qualified)
def test_model_class_declares_its_own_model_choice(model_cls):
    declared = vars(model_cls).get("model_choice")

    assert declared == EXPECTED_MODEL_CHOICE[model_cls], (
        f"{_qualified(model_cls)} declares model_choice={declared!r} in its own body, "
        f"expected {EXPECTED_MODEL_CHOICE[model_cls]!r}"
    )


def test_model_choices_are_unique():
    choices = list(EXPECTED_MODEL_CHOICE.values())
    duplicates = sorted({choice for choice in choices if choices.count(choice) > 1})

    assert not duplicates, f"model_choice values used by more than one class: {duplicates}"
