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

import warnings

from typing import Optional, Union



__all__ = (
    "SEVERITIES",
    "MEMBERS",
    "OpenDSMWarning",
    "ClusteringWarning",
    "StatsWarning",
)


SEVERITIES = ("infeasible", "disqualified", "advisory")
MEMBERS = ("data", "model", "settings")

_WARNING_REGISTRY: dict[str, type["OpenDSMWarning"]] = {}


class OpenDSMWarning(UserWarning):
    """A warning record, emitted through the warnings channel and serializable.

    Parameters
    ----------
    qualified_name : :any:`str`
        Dotted name identifying the check that produced the record, e.g.
        `'eemeter.sufficiency_criteria.no_data'`.
    description : :any:`str`
        Prose describing the nature of the warning.
    severity : :any:`str`
        One of `'infeasible'`, `'disqualified'`, `'advisory'`.
    data : :any:`dict` or :any:`list`, optional
        JSON-serializable evidence for why the record was produced. Never part of
        the emitted message.
    member : :any:`str`, optional
        Which part of the fit the record disqualifies: one of `'data'`, `'model'`,
        `'settings'`. Required when `severity` is `'disqualified'`, None otherwise.

    Notes
    -----
    Every field is passed to the base constructor as well as stored as an attribute,
    so the instance pickles and deep-copies through the default `BaseException`
    reduction, which rebuilds from `args`.
    """

    def __init__(
        self,
        qualified_name: str,
        description: str,
        severity: str,
        data: Optional[Union[dict, list]] = None,
        member: Optional[str] = None,
    ):
        if severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {severity!r}"
            )

        if severity == "disqualified" and member not in MEMBERS:
            raise ValueError(
                f"member must be one of {MEMBERS} when severity is 'disqualified', "
                f"got {member!r}"
            )

        if severity != "disqualified" and member is not None:
            raise ValueError(
                f"member must be None when severity is {severity!r}, got {member!r}"
            )

        super().__init__(qualified_name, description, severity, data, member)

        self.qualified_name = qualified_name
        self.description = description
        self.severity = severity
        self.data = data
        self.member = member

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _WARNING_REGISTRY[cls.__name__] = cls

    def __str__(self):
        return f"[{self.severity}] {self.qualified_name}: {self.description}"

    def __repr__(self):
        return f"{type(self).__name__}({self.qualified_name})"

    def json(self) -> dict:
        """Return a JSON-serializable representation of this record."""

        json_dict = {
            "category": type(self).__name__,
            "severity": self.severity,
            "member": self.member,
            "qualified_name": self.qualified_name,
            "description": self.description,
            "data": self.data,
        }

        return json_dict

    @classmethod
    def from_json(
        cls, json_dict: dict, legacy_severity: Optional[str] = None
    ) -> "OpenDSMWarning":
        """Rebuild a record from a `json()` payload, dispatching on its category.

        A payload carrying neither `category` nor `severity` is rebuilt as an
        `EEMeterWarning` with `legacy_severity`, which the caller knows from the list the
        entry was stored in.
        """

        category = json_dict.get("category")
        severity = json_dict.get("severity")

        if category is None or severity is None:
            category = "EEMeterWarning"
            severity = legacy_severity

        warning_cls = _WARNING_REGISTRY[category]
        warning = warning_cls(
            qualified_name=json_dict["qualified_name"],
            description=json_dict["description"],
            severity=severity,
            data=json_dict.get("data"),
            member=json_dict.get("member"),
        )

        return warning

    def warn(self):
        """Emit this record on the warnings channel."""

        warnings.warn(self, stacklevel=2)


_WARNING_REGISTRY[OpenDSMWarning.__name__] = OpenDSMWarning

# appended so a host policy already in the filter list (e.g. -W error) still wins; with no
# host policy this is the only entry matching OpenDSMWarning, so "always" still prevents a
# batch of meters producing the same text many times over from collapsing to one emission
warnings.filterwarnings("always", category=OpenDSMWarning, append=True)


class ClusteringWarning(OpenDSMWarning):
    """A record produced by the clustering utilities."""


class StatsWarning(OpenDSMWarning):
    """A record produced by the statistics utilities."""
