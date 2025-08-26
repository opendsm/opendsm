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

from abc import ABC, abstractmethod


class AbstractDataProcessor(ABC):
    """Abstract class for data processors."""

    @abstractmethod
    def set_data(self, data):
        """Process data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to process.

        Returns
        -------
        processed_data : pd.DataFrame
            Processed data.
        """
        pass

    @abstractmethod
    def _check_data_sufficiency(self, data):
        """Check data sufficiency.

        Parameters
        ----------
        data : pd.DataFrame
            Data to check.

        Returns
        -------
        is_sufficient : bool
            Whether the data is sufficient.
        """
        pass
