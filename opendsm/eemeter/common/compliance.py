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

from typing import Union

import pydantic

from opendsm.common.warnings import MEMBERS, OpenDSMWarning



__all__ = ("Compliance", "normalize_enforce_compliance")


_PRIOR_FORMAT_MEMBER_BY_QUALIFIED_NAME = {
    "eemeter.model_fit_metrics": "model",
    "eemeter.settings.nonstandard": "settings",
}


def _prior_format_member(payload: dict) -> str:
    """The `member` a prior-format disqualification payload belongs to.

    Derived from `qualified_name`: `eemeter.model_fit_metrics` and
    `eemeter.settings.nonstandard` are the only non-data qualified names prior-format
    payloads carry; every other qualified name was a data disqualification.
    """

    return _PRIOR_FORMAT_MEMBER_BY_QUALIFIED_NAME.get(payload.get("qualified_name"), "data")


def _prior_format_entries(info: dict) -> list[OpenDSMWarning]:
    """Rebuild the records a payload written before `schema_version` stored in its info block.

    Such a payload keeps its records in two lists whose severity is implied by the list.
    """

    entries = []
    for payload in info.get("disqualification") or []:
        entries.append(
            OpenDSMWarning.from_json(
                {"member": _prior_format_member(payload), **payload},
                legacy_severity="disqualified",
            )
        )

    for payload in info.get("warnings") or []:
        entries.append(OpenDSMWarning.from_json(payload, legacy_severity="advisory"))

    return entries


class Compliance(pydantic.BaseModel):
    """The full set of compliance records produced while fitting a model."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    all: list[OpenDSMWarning] = pydantic.Field(default_factory=list)

    def record(self, entry: OpenDSMWarning) -> None:
        """Append a record to `all`."""

        self.all.append(entry)

    @property
    def disqualification(self) -> list[OpenDSMWarning]:
        """Records with `severity == 'disqualified'`, in recorded order."""

        return [entry for entry in self.all if entry.severity == "disqualified"]

    @property
    def warnings(self) -> list[OpenDSMWarning]:
        """Records with `severity == 'advisory'`, in recorded order."""

        return [entry for entry in self.all if entry.severity == "advisory"]

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
