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

from opendsm.common.exceptions import OpenDSMError
from opendsm.common.warnings import OpenDSMWarning



__all__ = (
    "EEMeterError",
    "ComplianceError",
    "DataSufficiencyError",
    "InfeasibleDataError",
    "DisqualifiedModelError",
    "NonstandardSettingsError",
    "ModelFitError",
    "ModelLoadError",
    "NoBaselineDataError",
    "NoReportingDataError",
)


class EEMeterError(OpenDSMError):
    """Base class for EEmeter library errors."""


class ComplianceError(EEMeterError):
    """Base class for errors raised because compliance enforcement refused a fit or a
    prediction.

    Args:
        message: Why the fit or prediction was refused.
        entries: The compliance records that caused the refusal.

    Attributes:
        entries: Those records, as a list; empty when none were given. A caller looping
            over meters reads them to report why this one was refused and move on.
    """

    def __init__(self, message: str = "", entries: Optional[list] = None):
        super().__init__(message)
        self.entries: list[OpenDSMWarning] = list(entries or [])


class DataSufficiencyError(ComplianceError):
    """Error indicating insufficient data to fit model on."""


class InfeasibleDataError(DataSufficiencyError):
    """Error indicating data on which no model can be fit at all."""


class DisqualifiedModelError(ComplianceError):
    """Error indicating attempt to predict with disqualified or poorly fit model."""


class NonstandardSettingsError(ComplianceError):
    """Error indicating settings that depart from the compliant reference."""


class ModelFitError(EEMeterError):
    """Error indicating that a model could not be fit.

    Args:
        message: Why the fit failed.
        component: The part of the model that failed, kept as ``component``.
        segment: The portion of the data it was being fit on, kept as ``segment``.
    """

    def __init__(
        self,
        message: str = "",
        component: Optional[str] = None,
        segment: Optional[str] = None,
    ):
        super().__init__(message)
        self.component = component
        self.segment = segment


class ModelLoadError(EEMeterError):
    """Error indicating that a serialized model could not be rebuilt."""


class NoBaselineDataError(EEMeterError):
    """Error indicating lack of baseline data."""


class NoReportingDataError(EEMeterError):
    """Error indicating lack of reporting data."""
