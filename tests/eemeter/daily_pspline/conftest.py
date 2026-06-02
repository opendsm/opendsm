"""Shared fixtures for daily_pspline tests."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def v_shaped_data(rng):
    """V-shaped energy-temperature data: heating + baseload + cooling."""
    T = np.sort(rng.uniform(20, 90, 200))
    y = 30 + 0.5 * np.maximum(55 - T, 0) + 0.3 * np.maximum(T - 70, 0) + rng.normal(0, 1, 200)
    return T, y


@pytest.fixture
def heating_only_data(rng):
    """Heating-only meter (gas): monotone decreasing energy vs temperature."""
    T = np.sort(rng.uniform(10, 70, 150))
    y = 50 + 0.8 * np.maximum(60 - T, 0) + rng.normal(0, 0.5, 150)
    return T, y


@pytest.fixture
def flat_data(rng):
    """Temperature-independent (flat) energy."""
    T = np.sort(rng.uniform(30, 80, 100))
    y = 20 + rng.normal(0, 0.3, 100)
    return T, y


@pytest.fixture
def dev_settings():
    """Developer-mode settings for testing (suppresses warning)."""
    return DailyPSplineSettings(developer_mode=True, silent_developer_mode=True)
