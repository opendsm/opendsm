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
from typing import Union
import pydantic

__all__ = ("EEMeterWarning",)


class EEMeterWarning(pydantic.BaseModel):
    """An object representing a warning and data associated with it.

    Attributes
    ----------
    qualified_name : :any:`str`
        Qualified name, e.g., `'eemeter.method_abc.missing_data'`.
    description : :any:`str`
        Prose describing the nature of the warning.
    data : :any:`dict`
        Data that reproducibly shows why the warning was issued. Data should
        be JSON serializable.
    """

    qualified_name: str
    description: str
    data: Union[dict, list]

    def __repr__(self):
        return "EEMeterWarning(qualified_name={})".format(self.qualified_name)

    def __str__(self):
        return repr(self)

    def json(self) -> dict:
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "qualified_name": self.qualified_name,
            "description": self.description,
            "data": self.data,
        }

    def warn(self):
        data = ""
        if self.data:
            data = f"\n{self.data}"
        logging.getLogger("eemeter").warning(f"{self.description}{data}")
