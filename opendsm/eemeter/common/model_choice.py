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

from opendsm.eemeter.common.exceptions import ModelLoadError



def check_model_choice(payload: dict, expected: str) -> None:
    """Reject a serialized model payload written by another model class.

    Args:
        payload: A model's `to_dict()` output.
        expected: The `model_choice` of the class loading the payload.

    Raises:
        ModelLoadError: If the payload's `model_choice` differs from `expected`. A payload
            without `model_choice` passes.
    """
    found = payload.get("model_choice")
    if found is not None and found != expected:
        raise ModelLoadError(
            f"Payload was written by the {found!r} model; this class loads {expected!r} payloads"
        )
