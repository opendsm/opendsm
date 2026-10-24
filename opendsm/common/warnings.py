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
        # `from_json` dispatches on the bare class name, so a second class under one name
        # would silently take over the first's payloads
        registered = _WARNING_REGISTRY.get(cls.__name__)
        if registered is not None and registered is not cls:
            raise TypeError(
                f"An OpenDSMWarning subclass named {cls.__name__!r} is already registered "
                f"as {registered.__module__}.{registered.__qualname__}"
            )

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
    def from_json(cls, json_dict: dict) -> "OpenDSMWarning":
        """Rebuild a record from a `json()` payload, dispatching on its category.

        Raises `KeyError` when the payload lacks `category`, `severity`, `qualified_name`
        or `description`, or names a category no `OpenDSMWarning` subclass registered.
        """

        warning_cls = _WARNING_REGISTRY[json_dict["category"]]
        warning = warning_cls(
            qualified_name=json_dict["qualified_name"],
            description=json_dict["description"],
            severity=json_dict["severity"],
            data=json_dict.get("data"),
            member=json_dict.get("member"),
        )

        return warning

    def warn(self, stacklevel: int = 2):
        """Emit this record on the warnings channel.

        `stacklevel` counts frames as `warnings.warn` does, with this method as frame 1;
        the default attributes the warning to the caller of `warn`.
        """

        warnings.warn(self, stacklevel=stacklevel)

_WARNING_REGISTRY[OpenDSMWarning.__name__] = OpenDSMWarning

# appended so a host policy such as -W error still wins; "always" keeps identical records
# from a batch of meters from collapsing to one emission
warnings.filterwarnings("always", category=OpenDSMWarning, append=True)


class ClusteringWarning(OpenDSMWarning):
    """A record produced by the clustering utilities."""


class StatsWarning(OpenDSMWarning):
    """A record produced by the statistics utilities."""
