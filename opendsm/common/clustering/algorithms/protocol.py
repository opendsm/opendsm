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

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from opendsm.common.clustering.metrics.labels import ClusteringResult


class ClusterAlgorithm(Protocol):
    """Protocol for all clustering algorithm functions.

    Each algorithm receives the feature data and the full ClusteringSettings.
    The algorithm extracts its own sub-settings, seed, min_cluster_size,
    small_cluster_mode, etc. from the settings object.

    Enforcing the return type here ensures a misimplemented algorithm
    fails at import/type-check time rather than deep in the scoring pipeline.
    """

    def __call__(
        self,
        data: np.ndarray,
        settings: Any,
    ) -> ClusteringResult: ...
