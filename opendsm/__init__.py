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

import logging as _logging

from importlib.metadata import metadata, PackageNotFoundError

try:
    _meta = metadata("opendsm")
except PackageNotFoundError:
    _meta = {}

__title__ = _meta.get("Name", "opendsm")
__version__ = _meta.get("Version", "unknown")
__description__ = _meta.get("Summary", "")
__author__ = _meta.get("Author", "")
__author_email__ = _meta.get("Author-email", "")
__license__ = _meta.get("License", "")
__url__ = "http://github.com/opendsm/opendsm"
__copyright__ = "Copyright 2014-2025 OpenDSM contributors"

import platform
import warnings

# these happen during native code execution and segfault pytest when filterwarnings is set to error
warnings.filterwarnings("ignore", module="importlib._bootstrap")
warnings.filterwarnings(
    "ignore", "builtin type swigvarlink has no __module__ attribute"
)
warnings.filterwarnings(
    "ignore", "builtin type SwigPyPacked has no __module__ attribute"
)

if platform.system() == "Windows":
    # numba JIT breaks on Windows with int32/int64 return types
    from numba import config

    config.DISABLE_JIT = True

from .common import test_data
from . import (
    eemeter,
    drmeter,
    comparison_groups,
)

# Set default logging handler to avoid "No handler found" warnings.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# exclude built-in imports from namespace
__all__ = [
    "__title__",
    "__description__",
    "__url__",
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "eemeter",
    "drmeter",
    "comparison_groups",
    "test_data",
]

def __dir__():
    return __all__