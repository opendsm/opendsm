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

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.savings.correction import (
    cg_member_ids,
    correct_reporting,
)
from opendsm.comparison_groups.savings.savings import compute_savings
from opendsm.comparison_groups.savings.settings import CGCorrectionSettings



def _scoped(ledger, ids):
    """The ledger rows whose meter id is in ``ids``."""
    rows = ledger[ledger["id"].isin(ids)]

    return rows


class MeterAnalysis:
    """Correction and savings for ONE treatment meter against an already-chosen
    comparison group.

    ``select_comparison_group`` is the group-level entry point and runs once for
    the whole treatment population; the selection it returns is passed here per
    meter. The stages run in order: ``correct()`` corrects this meter's
    reporting-period model with its comparison group, ``savings()`` reduces the
    correction to avoided energy, and ``run()`` chains both. Each stage stores
    its result and returns ``self`` so calls chain; a ``correct()`` that raises
    clears both stored results first, so nothing stale survives a failure.

    A meter that cannot be corrected at all raises ``MeterCorrectionError``;
    individual timesteps that cannot be corrected come back as NaN rows instead.
    Uncertainty columns downstream are the models' heuristic combined bands, not
    calibrated intervals.

    ``treatment``/``pool`` carry their own ``baseline_window``/``reporting_window``,
    so the facade reads those directly rather than accepting window params.
    ``min_window_coverage`` lives solely on ``correction_settings``
    (``CGCorrectionSettings.min_window_coverage``; the package default applies
    when no settings are supplied) and governs the reporting-coverage prune.
    """

    def __init__(self, selection, treatment, pool, treatment_id, correction_settings=None):
        if correction_settings is None:
            correction_settings = CGCorrectionSettings()

        self.selection = selection
        self.treatment = treatment
        self.pool = pool
        self.treatment_id = str(treatment_id)
        self.correction_settings = correction_settings
        self.correction = None
        self.savings_result = None
        self.guard_exclusions = exclusions.empty_ledger()

    # -- stages --------------------------------------------------------------

    def correct(self, period=None, prior=None):
        """Correct this meter's reporting-period model with its comparison group.

        Args:
            period: optional ``(start, end)`` bounds restricting the reported
                window.
            prior: optional earlier ``CorrectionResult`` for this meter; its
                comparison group and read periods are reused.

        Raises:
            MeterCorrectionError: the meter cannot be corrected; its ledger rows
                are kept for ``meter_log()``.
        """
        self.correction = None
        self.savings_result = None
        self.guard_exclusions = exclusions.empty_ledger()

        try:
            correction = correct_reporting(
                self.selection,
                self.treatment,
                self.pool,
                self.treatment_id,
                settings=self.correction_settings,
                period=period,
                prior=prior,
            )
        except exclusions.MeterCorrectionError as error:
            # A raising guard leaves no CorrectionResult, so its rows would be
            # unreachable from meter_log() — the one place that explains the failure.
            self.guard_exclusions = error.exclusions
            raise

        self.correction = correction

        return self

    def savings(self, observed_unc=None, aggregation="native", season_def=None):
        """Reduce the stored correction to avoided energy.

        Args:
            observed_unc: optional observed-uncertainty override, as accepted by
                ``compute_savings``.
            aggregation: ``"native"``, ``"monthly"``, ``"seasonal"``, ``"annual"``
                or ``"total"``.
            season_def: month-name to season mapping; the package default when
                omitted.

        Raises:
            RuntimeError: ``correct()`` has not run.
        """
        if self.correction is None:
            raise RuntimeError("run correct() first before savings().")

        savings = compute_savings(
            self.correction,
            observed_unc=observed_unc,
            aggregation=aggregation,
            season_def=season_def,
        )
        self.savings_result = savings

        return self

    def run(self, period=None, prior=None, observed_unc=None, aggregation="native", season_def=None):
        """Run ``correct()`` then ``savings()`` with the same arguments."""
        self.correct(period=period, prior=prior)
        self.savings(observed_unc=observed_unc, aggregation=aggregation, season_def=season_def)

        return self

    def meter_log(self):
        """The merged disqualification ledger for this meter and its comparison-
        group members, in the shared ``[id, stage, origin, reason, detail]``
        schema, ordered population, selection, correction, then meter id.

        Scope is this analysis: the treatment meter's own rows plus pool-side
        rows for the meters in its comparison group, so a drop that starved the
        group is visible without dragging in unrelated pool meters. The
        population-wide ledgers remain readable off the population objects.
        Membership is the raw selection walk, which still covers meters the
        correction dropped — those are exactly the rows this log is for.
        Correction rows appear once ``correct()`` has run, including when it
        raised.
        """
        member_ids = set(cg_member_ids(self.selection, self.treatment_id))
        scope = member_ids | {self.treatment_id}
        ledgers = [
            _scoped(self.treatment.exclusions, {self.treatment_id}),
            _scoped(self.pool.exclusions, member_ids),
            _scoped(self.selection.exclusions, scope),
            _scoped(self.guard_exclusions, scope),
        ]

        if self.correction is not None:
            ledgers.append(_scoped(self.correction.exclusions, scope))

        merged = exclusions.merge(*ledgers)

        return merged
