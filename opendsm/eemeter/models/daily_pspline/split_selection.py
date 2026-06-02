"""Season / weekday-weekend split generation, filtering, and segment extraction.

Generates candidate split combinations, trims by data sufficiency and
ellipsoid overlap, and provides segment filtering for component keys.
Designed for reuse across model types.
"""

from __future__ import annotations

import itertools

import pandas as pd

from opendsm.eemeter.models.daily.utilities.ellipsoid_test import ellipsoid_split_filter

SEASONS = {"su": "summer", "sh": "shoulder", "wi": "winter"}
DAYS = {"fw": ["weekday", "weekend"], "wd": ["weekday"], "we": ["weekend"]}

_SEASONAL_OPTIONS = [
    ["su_sh_wi"],
    ["su", "sh_wi"],
    ["su_sh", "wi"],
    ["su_wi", "sh"],
    ["su", "sh", "wi"],
]
_DAY_OPTIONS = [["wd", "we"]]


def segment(component: str, df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame rows matching a component key like 'wd-su' or 'fw-su_sh_wi'."""
    day_key = component[:2]
    season_keys = component[3:].split("_")
    days = DAYS[day_key]
    seasons = [SEASONS[k] for k in season_keys]
    return df[df["season"].isin(seasons) & df["weekday_weekend"].isin(days)]


class SplitSelector:
    """Generates, filters, and extracts season/day split combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Meter data with 'season', 'weekday_weekend', 'temperature', 'observed'.
    split_settings
        Split-selection settings with allow_separate_*, reduce_splits_by_gaussian, etc.
    split_min_days : int
        Minimum observations required per season to allow a separate split.
    """

    def __init__(self, df: pd.DataFrame, split_settings, split_min_days: int = 30):
        self._df = df
        self._ss = split_settings
        self._split_min_days = split_min_days

    def combinations(self) -> list[str]:
        """Valid season × day combination strings for the current data."""
        if hasattr(self._ss, 'allow_splits') and not self._ss.allow_splits:
            return ["fw-su_sh_wi"]
        raw = self._generate()
        return self._trim(raw)

    def required_components(self, combinations: list[str]) -> list[str]:
        """Union of all component keys; always includes 'fw-su_sh_wi' baseline."""
        needed: set[str] = {"fw-su_sh_wi"}
        for combo in combinations:
            needed.update(combo.split("__"))
        return sorted(needed, key=lambda x: (len(x), x))

    def segment(self, component: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter rows matching a component key like 'wd-su' or 'fw-su_sh_wi'."""
        return segment(component, df if df is not None else self._df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _generate() -> list[str]:
        combos: set[str] = set()
        for day_types in _DAY_OPTIONS:
            for season_partition in _SEASONAL_OPTIONS:
                for choices in itertools.product(["fw", "split"], repeat=len(season_partition)):
                    fw_parts: list[str] = []
                    day_parts: dict[str, list[str]] = {dt: [] for dt in day_types}
                    for season_group, choice in zip(season_partition, choices):
                        if choice == "fw":
                            fw_parts.append(f"fw-{season_group}")
                        else:
                            for day_type in day_types:
                                day_parts[day_type].append(f"{day_type}-{season_group}")
                    components = fw_parts + [c for dt in day_types for c in day_parts[dt]]
                    combos.add("__".join(components))
        return sorted(combos, key=lambda x: (len(x), x))

    def _trim(self, combo_list: list[str]) -> list[str]:
        meter = self._df
        ss = self._ss
        min_days = self._split_min_days
        we_days = DAYS["we"]

        allow = {
            "su": ss.allow_separate_summer and (meter["season"] == "summer").sum() >= min_days,
            "sh": ss.allow_separate_shoulder and (meter["season"] == "shoulder").sum() >= min_days,
            "wi": ss.allow_separate_winter and (meter["season"] == "winter").sum() >= min_days,
        }
        allow_wd = ss.allow_separate_weekday_weekend

        if ss.reduce_splits_by_gaussian:
            gaussian = ellipsoid_split_filter(meter, n_std=ss.reduce_splits_num_std)
            allow = {k: allow[k] and gaussian[SEASONS[k]] for k in allow}
            allow_wd = allow_wd and gaussian["weekday_weekend"]

        we_counts = {
            s: ((meter["season"] == SEASONS[s]) & meter["weekday_weekend"].isin(we_days)).sum()
            for s in ("su", "sh", "wi")
        }
        we_min = min_days / 3.75

        trimmed = []
        for combo in combo_list:
            if "wd" in combo and not allow_wd:
                continue
            valid = True
            for component in combo.split("__"):
                season_keys = component[3:].split("_")
                if len(season_keys) == 1 and not allow.get(season_keys[0], True):
                    valid = False
                    break
                if sum(we_counts[s] for s in season_keys) < we_min:
                    valid = False
                    break
            if valid:
                trimmed.append(combo)
        return trimmed
