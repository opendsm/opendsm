#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Collection, Union

import pydantic

from opendsm.common.warnings import MEMBERS, OpenDSMWarning
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning



__all__ = (
    "SCHEMA_VERSION",
    "Compliance",
    "check_schema_version",
    "normalize_enforce_compliance",
    "prior_format_entries",
    "raise_if_baseline_disqualified",
    "raise_if_infeasible",
    "raise_if_model_disqualified",
    "record_timezone_mismatch",
)


SCHEMA_VERSION = 1


_PRIOR_FORMAT_MEMBER_BY_QUALIFIED_NAME = {
    "eemeter.model_fit_metrics": "model",
    "eemeter.settings.nonstandard": "settings",
}


def check_schema_version(v: int) -> int:
    """Return a serialized model payload's `schema_version` unchanged.

    Raises `ValueError` when it is newer than `SCHEMA_VERSION`, the newest version this
    package reads.
    """

    if v > SCHEMA_VERSION:
        raise ValueError(
            f"payload schema_version {v} is newer than the {SCHEMA_VERSION} this "
            "package can read"
        )

    return v


def prior_format_entries(info: dict) -> list[EEMeterWarning]:
    """Rebuild the records a payload without `schema_version` keeps in its info block.

    Such a payload keeps its records in two lists, `disqualification` and `warnings`, whose
    severity is implied by the list. A disqualification's `member` is derived from its
    `qualified_name`: `eemeter.model_fit_metrics` and `eemeter.settings.nonstandard` are the
    only non-data qualified names these payloads carry; every other one is a data
    disqualification. Raises `KeyError` when an entry lacks `qualified_name` or
    `description`.
    """

    entries = []
    for payload in info.get("disqualification") or []:
        entries.append(
            EEMeterWarning(
                qualified_name=payload["qualified_name"],
                description=payload["description"],
                severity="disqualified",
                data=payload.get("data"),
                member=_PRIOR_FORMAT_MEMBER_BY_QUALIFIED_NAME.get(
                    payload["qualified_name"], "data"
                ),
            )
        )

    for payload in info.get("warnings") or []:
        entries.append(
            EEMeterWarning(
                qualified_name=payload["qualified_name"],
                description=payload["description"],
                severity="advisory",
                data=payload.get("data"),
            )
        )

    return entries


class Compliance(pydantic.BaseModel):
    """The compliance records produced while checking data or fitting a model."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    all: list[OpenDSMWarning] = pydantic.Field(default_factory=list)

    def record(self, entry: OpenDSMWarning) -> None:
        """Append a record to `all`."""

        self.all.append(entry)

    def record_and_warn(self, entry: OpenDSMWarning) -> None:
        """Append a record to `all`, then emit it on the warnings channel.

        A record is emitted once, where it is first recorded; a reader of stored records,
        such as a model fit on the data that recorded them, never re-emits them.
        """

        self.record(entry)
        entry.warn(stacklevel=3)

    @property
    def disqualification(self) -> list[OpenDSMWarning]:
        """Records with `severity == 'disqualified'`, in recorded order."""

        return [entry for entry in self.all if entry.severity == "disqualified"]

    @property
    def warnings(self) -> list[OpenDSMWarning]:
        """Records with `severity == 'advisory'`, in recorded order."""

        return [entry for entry in self.all if entry.severity == "advisory"]

    @property
    def infeasible(self) -> list[OpenDSMWarning]:
        """Records with `severity == 'infeasible'`, in recorded order."""

        return [entry for entry in self.all if entry.severity == "infeasible"]

    def disqualified_in(self, members: Collection[str]) -> list[OpenDSMWarning]:
        """The `disqualification` records whose `member` is in `members`, in recorded order."""

        return [entry for entry in self.disqualification if entry.member in members]

    @property
    def status(self) -> list[str]:
        """The sorted, deduplicated set of `member` values over `disqualification`."""

        return sorted({entry.member for entry in self.disqualification})

    @property
    def qualified(self) -> bool:
        """True when `status` is empty."""

        return not self.status


def normalize_enforce_compliance(enforce_compliance: Union[bool, list, tuple]) -> frozenset:
    """Normalize the `enforce_compliance` argument to a frozenset of members.

    `True` enforces all members, `False` enforces none, and a list or tuple names the
    members to enforce; each name must be one of `opendsm.common.warnings.MEMBERS`.
    Raises `ValueError` for a name outside `MEMBERS` and `TypeError` for any other
    argument type.
    """

    if isinstance(enforce_compliance, bool):
        if enforce_compliance:
            return frozenset(MEMBERS)

        return frozenset()

    if isinstance(enforce_compliance, (list, tuple)):
        for name in enforce_compliance:
            if name not in MEMBERS:
                raise ValueError(
                    f"enforce_compliance member must be one of {MEMBERS}, got {name!r}"
                )

        return frozenset(enforce_compliance)

    raise TypeError(
        f"enforce_compliance must be a bool, list, or tuple, got {type(enforce_compliance)!r}"
    )


def raise_if_infeasible(compliance: Compliance) -> None:
    """Raise `InfeasibleDataError` carrying every infeasible record, if there is one."""

    entries = compliance.infeasible
    if entries:
        raise InfeasibleDataError("Can't fit any model on this baseline data", entries=entries)


def raise_if_baseline_disqualified(compliance: Compliance, enforced: frozenset) -> None:
    """Raise `DataSufficiencyError` when 'data' is in `enforced` and the baseline records
    include a data disqualification; the error carries those records."""

    if "data" not in enforced:
        return

    entries = compliance.disqualified_in({"data"})
    if entries:
        raise DataSufficiencyError("Can't fit model on disqualified baseline data", entries=entries)


def raise_if_model_disqualified(compliance: Compliance, enforced: frozenset) -> None:
    """Raise `DisqualifiedModelError` carrying a fitted model's disqualifications whose
    member is in `enforced`, if there are any."""

    entries = compliance.disqualified_in(enforced)
    if entries:
        raise DisqualifiedModelError(
            "Attempting to predict using disqualified model without relaxing enforce_compliance",
            entries=entries,
        )


def record_timezone_mismatch(compliance, baseline_timezone, reporting_timezone) -> None:
    """Record and emit a disqualified data entry on `compliance` when reporting data is not
    in the model's timezone.

    Timezones are compared as strings to accommodate mixed tzinfo implementations.
    """

    if str(baseline_timezone) == str(reporting_timezone):
        return

    entry = EEMeterWarning(
        qualified_name="eemeter.data_quality.timezone_mismatch",
        description=(
            "Reporting data is in a different timezone than the model was fit on, so its "
            "period boundaries do not line up with the baseline."
        ),
        severity="disqualified",
        member="data",
        data={
            "baseline_timezone": str(baseline_timezone),
            "reporting_timezone": str(reporting_timezone),
        },
    )
    compliance.record_and_warn(entry)
