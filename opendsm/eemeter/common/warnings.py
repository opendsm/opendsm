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

from typing import Optional

from opendsm.common.warnings import OpenDSMWarning



__all__ = ("EEMeterWarning", "nonstandard_settings_warning")


class EEMeterWarning(OpenDSMWarning):
    """A record produced by the eemeter models and their data checks."""


def nonstandard_settings_warning(deviations: dict) -> Optional[EEMeterWarning]:
    """Return a warning carrying a settings deviation report.

    Args:
        deviations: A report built by `settings_deviation_report`.

    Returns:
        An `eemeter.settings.nonstandard` warning whose `data["deviations"]` is
        `deviations`, or None when that report is empty.
    """
    if not deviations:
        return None

    warning = EEMeterWarning(
        qualified_name="eemeter.settings.nonstandard",
        description=(
            "Model settings deviate from the OpenDSM defaults. Results are not standard "
            "OpenDSM outputs."
        ),
        severity="disqualified",
        member="settings",
        data={"deviations": deviations},
    )

    return warning
