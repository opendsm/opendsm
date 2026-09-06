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

import logging
from typing import Optional, Union
import pydantic

from opendsm.common.base_settings import BaseSettings, settings_deviations



__all__ = ("EEMeterWarning", "nonstandard_settings_warning")


class EEMeterWarning(pydantic.BaseModel):
    """An object representing a warning and data associated with it.

    Attributes
    ----------
    qualified_name : :any:`str`
        Qualified name, e.g., `'eemeter.method_abc.missing_data'`.
    description : :any:`str`
        Prose describing the nature of the warning.
    data : :any:`dict` or :any:`list`, optional
        Data that reproducibly shows why the warning was issued. Data should
        be JSON serializable. Defaults to None.
    """

    qualified_name: str
    description: str
    data: Optional[Union[dict, list]] = None

    def __repr__(self):
        return "EEMeterWarning(qualified_name={})".format(self.qualified_name)

    def __str__(self):
        return repr(self)

    def json(self) -> dict:
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        json_dict = {
            "qualified_name": self.qualified_name,
            "description": self.description,
            "data": self.data,
        }

        return json_dict

    def warn(self):
        data = ""
        if self.data:
            data = f"\n{self.data}"
        logging.getLogger("eemeter").warning(f"{self.description}{data}")


def nonstandard_settings_warning(
    settings: BaseSettings, reference: BaseSettings
) -> Optional[EEMeterWarning]:
    """Return a warning describing any developer-tier settings that deviate from a reference.

    Returns None when `settings` matches `reference` on every developer-tier field.
    """

    deviations = settings_deviations(settings, reference)
    if not deviations:
        return None

    deviation_data = {
        path: {"value": value, "default": default} for path, value, default in deviations
    }
    warning = EEMeterWarning(
        qualified_name="eemeter.settings.nonstandard",
        description=(
            "Model settings deviate from the OpenDSM defaults. Results are not standard "
            "OpenDSM outputs."
        ),
        data={"deviations": deviation_data},
    )

    return warning
