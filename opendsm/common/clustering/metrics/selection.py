"""Label selection helpers for the Labels container.

Extracted from Labels to keep the container thin.  All functions
are pure (no mutation) — they take data in and return a result.
"""

from __future__ import annotations

import types
import warnings

import numpy as np

from opendsm.common.clustering.metrics.single_k_metrics import SingleKMetrics
from opendsm.common.clustering.metrics import voting as _voting


def _compute_composite_score(
    lm: SingleKMetrics | None,
    council: dict[str, float],
    extra_scores: dict[str, float] | None = None,
) -> types.SimpleNamespace:
    """Build a ``SimpleNamespace(score=...)`` proxy for Schulze voting.

    Score semantics
    ---------------
    np.nan  : voter abstains — skips all pairwise comparisons for this candidate
    np.inf  : active worst-score vote — voter ranks this candidate last
    -np.inf : active best-score vote — voter ranks this candidate first
    finite  : normal score value

    For invalid labelings (*lm* is ``None``), single-k scores default to
    ``np.inf`` (active worst vote).  Cross-k scores from *extra_scores* are
    injected as-is and may be NaN (abstain) or a valid score value.
    """
    # Default: inf (active worst) for single-k, will be overridden by extra_scores
    score: dict[str, float] = {m: np.inf for m in council}

    # Inject cross-k (or any supplemental) scores first; preserves nan/inf/-inf
    if extra_scores:
        for metric, s in extra_scores.items():
            if metric in score:
                score[metric] = s

    if lm is None:
        return types.SimpleNamespace(score=score)

    # Single-k attributes for valid lm
    for metric, weight in council.items():
        if weight <= 0:
            continue
        if extra_scores and metric in extra_scores:
            continue  # already set from extra_scores
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                s = getattr(lm, metric)
            # np.nan → abstain; inf/-inf/finite → pass through as-is
            score[metric] = float(s)  # also converts numpy scalars
        except Exception:
            score[metric] = np.nan  # uncomputable → abstain, not penalise

    # Outlier-fraction penalty: divide finite scores by coverage so labelings
    # that discard many samples rank worse.  coverage=1.0 → no effect.
    # _coverage is set to 1.0 when outlier_fraction_penalty=False.
    coverage = getattr(lm, '_coverage', 1.0)
    if coverage < 1.0 - 1e-10:
        for metric in score:
            val = score[metric]
            if np.isfinite(val):
                score[metric] = val / coverage

    return types.SimpleNamespace(score=score)


def select_best_within_k(
    lms: list[SingleKMetrics],
    council: dict[str, float],
) -> SingleKMetrics:
    """Pick best labeling via weighted majority vote across per-k indices.

    All indices are oriented lower-is-better.  Each metric's vote is
    weighted by its council weight.

    Score semantics (consistent with the Schulze pipeline):
    - np.nan  : voter abstains — metric skips this labeling
    - np.inf  : active worst vote — labeling ranks last for this metric
    - -np.inf : active best vote — labeling ranks first for this metric
    """
    wins = np.zeros(len(lms))
    for metric, weight in council.items():
        if weight <= 0:
            continue
        scores = []
        for lm in lms:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    s = float(getattr(lm, metric))
                scores.append(s)
            except Exception:
                scores.append(np.nan)  # uncomputable → abstain
        arr = np.array(scores, dtype=float)
        # Skip this metric if all labelings abstain
        if np.all(np.isnan(arr)):
            continue
        # nanargmin ignores NaN (abstaining labelings) and treats inf as worst
        best_idx = int(np.nanargmin(arr))
        wins[best_idx] += weight
    if wins.max() == 0:
        return min(lms, key=lambda lm: lm._WCSS)
    return lms[int(np.argmax(wins))]


def _discriminability_weights(
    score_matrix: np.ndarray,
    voter_names: list[str],
    council: dict[str, float],
) -> dict[str, float]:
    """Scale council weights by each voter's discriminability on this dataset.

    Discriminability = coefficient of variation of finite scores across
    candidates.  A voter whose scores are nearly flat across all k values
    has low discriminability and gets downweighted.  A voter with a sharp
    minimum (clear preference) keeps its full weight.

    The scaling preserves the original weight ratios when all voters are
    equally discriminating, and attenuates flat voters toward zero.
    """
    adjusted = dict(council)
    for v_idx, name in enumerate(voter_names):
        if name not in council or council[name] <= 0:
            continue
        col = score_matrix[:, v_idx]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            continue
        mean = np.mean(finite)
        std = np.std(finite)
        # Coefficient of variation (scale-free spread measure).
        # High CV = scores vary a lot across k → good discriminability.
        # Low CV = flat profile → poor discriminability.
        cv = std / abs(mean) if abs(mean) > 1e-10 else std
        # Sigmoid mapping: cv → [0, 1] with midpoint at cv=0.1
        # Below cv=0.01 the voter is essentially flat → near-zero weight.
        discriminability = float(cv / (cv + 0.1))
        adjusted[name] = council[name] * discriminability
    return adjusted


def select_best_across_k(
    candidates: list[SingleKMetrics | None],
    council: dict[str, float],
    window_size: float,
    extra_scores_list: list[dict[str, float]] | None = None,
    candidate_k_values: list[int] | None = None,
    k_penalty_strength: float = 1.0,
    k_penalty_rate: float = 1.0,
) -> tuple[int, float]:
    """Schulze vote over a flat list of candidates.

    *candidates* may contain ``None`` entries for invalid labelings
    (they participate as all-inf so Gaussian smoothing sees the full
    candidate set in order).

    *extra_scores_list* is a parallel list of supplemental score dicts,
    one per candidate.  Used to inject cross-k index scores.  A NaN value
    means the voter abstains for that candidate.

    *score_multipliers* is a parallel list of per-candidate multipliers
    applied to all scores before voting.  Used for gap-based k-penalty:
    k=2 gets a small multiplier (scores become less favorable), higher k
    gets multipliers closer to 1 (no change).

    Returns
    -------
    winner_idx : int
        Winning index into *candidates*.
    confidence : float
        Selection confidence in [0, 1] from Schulze voting.
    """
    if extra_scores_list is None:
        extra_scores_list = [None] * len(candidates)

    proxies = [
        _compute_composite_score(c, council, extra)
        for c, extra in zip(candidates, extra_scores_list)
    ]

    score_matrix, voter_names, abstain_mask = _voting.build_rank_matrix(proxies)
    return _voting.schulze_voting(
        score_matrix, voter_names, voter_weights=council, window_size=window_size,
        candidate_k_values=candidate_k_values,
        k_penalty_strength=k_penalty_strength,
        k_penalty_rate=k_penalty_rate,
    )
