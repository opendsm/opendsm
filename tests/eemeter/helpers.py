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

"""Helpers shared by the eemeter model tests."""


def nonstandard_settings_warnings(warnings):
    """Return the warnings in `warnings` whose qualified name is
    'eemeter.settings.nonstandard'."""
    matching = [w for w in warnings if w.qualified_name == "eemeter.settings.nonstandard"]

    return matching
