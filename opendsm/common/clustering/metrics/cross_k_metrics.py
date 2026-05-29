"""Cross-k cluster validity indices and null tests.

Cross-k indices compare within-cluster sum-of-squares (WCSS) across
multiple *k* values to identify the optimal number of clusters.

Null tests determine whether any cluster structure exists at all.
Each null test returns a **p-value** (small p = strong evidence of
cluster structure).  The ``has_cluster_structure`` helper combines
them via grouped agreement.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pydantic
from scipy import stats
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    computed_field_cached_property,
)


class CrossKMetrics(ArbitraryPydanticModel):
    """Cross-k cluster validity indices and null tests.

    Constructed from WCSS values across different k.  Indices and null
    tests are lazily computed and cached, matching the ``SingleKMetrics``
    pattern.

    Null tests
    ----------
    Each null test returns a p-value: small p = strong evidence of
    cluster structure.

    >>> ckm = CrossKMetrics(wcss_by_k={2: 100, 3: 60, 4: 50}, ...)
    >>> ckm.gap_statistic          # float p-value
    >>> ckm.hopkins_test           # float p-value
    >>> ckm.sigclust_test          # float p-value
    >>> ckm.spectral_gap_test      # float p-value
    """

    wcss_by_k: dict[int, float] = pydantic.Field(
        description="Within-cluster sum of squares keyed by k.",
    )

    n_scored_by_k: dict[int, int] = pydantic.Field(
        default_factory=dict,
        description="Number of scored (non-outlier) samples per k. "
                    "Used to normalize WCSS for fair comparison against "
                    "null references that use all n samples. Empty dict "
                    "means all k use n_samples (no outlier removal).",
    )

    k_values: list[int] = pydantic.Field(
        description="Sorted list of k values present in the store.",
    )

    n_features: int = pydantic.Field(
        description="Number of features (columns) in the data.",
    )

    n_samples: int = pydantic.Field(
        description="Number of samples (rows) in the data.",
    )

    raw_data: np.ndarray | None = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
        description="Original (untransformed) data for null tests (gap "
                    "statistic, Hopkins, SigClust, spectral gap). Using raw "
                    "data prevents transform artifacts (variance cap, wavelet "
                    "normalization) from creating false cluster structure. "
                    "None disables data-dependent null tests.",
    )

    seed: int = pydantic.Field(default=42)

    _eps: float = 1e-10
    _n_gap_references: int = 5
    # Maximum number of k values to test in null tests.  The gate is binary
    # (structure or not), so testing a sparse subset of k is sufficient.
    # Set to 0 or None to test all k values.
    _max_null_test_k: int = 4

    def _sparse_k_vals(self, all_k_vals: list[int]) -> list[int]:
        """Select a sparse subset of k values for null tests.

        When _max_null_test_k is set and len(all_k_vals) exceeds it,
        returns an evenly-spaced subset including the first and last k.
        This reduces null test cost by ~5x while preserving sensitivity
        at multiple scales.
        """
        if not self._max_null_test_k or len(all_k_vals) <= self._max_null_test_k:
            return all_k_vals
        n = self._max_null_test_k
        indices = np.linspace(0, len(all_k_vals) - 1, n, dtype=int)
        return [all_k_vals[i] for i in indices]

    # ------------------------------------------------------------------
    # Index discovery
    # ------------------------------------------------------------------

    @classmethod
    def available_indices(cls) -> list[str]:
        """Return names of all *_index computed properties.

        This is the single source of truth for valid cross-k metric names.
        """
        return sorted(name for name in dir(cls) if name.endswith('_index'))

    # ------------------------------------------------------------------
    # Shared data statistics (cached, used by multiple null tests)
    # ------------------------------------------------------------------

    @cached_property
    def _bbox(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding box: (lo, hi, spread). Shared by gap + hopkins."""
        lo = self.raw_data.min(axis=0)
        hi = self.raw_data.max(axis=0)
        spread = np.where((hi - lo) < 1e-10, 1.0, hi - lo)
        return lo, hi, spread

    @cached_property
    def _cov_eigvals(self) -> np.ndarray:
        """Correlation-matrix eigenvalues, clipped to >= 0.

        Uses the correlation matrix (standardized covariance) instead of
        the raw covariance.  This prevents magnitude-dominated features
        from capturing all variance in the first eigenvalue, which would
        hide shape-based cluster structure from sigclust and spectral gap.
        """
        # Standardize per-feature to unit variance → covariance = correlation
        std = np.std(self.raw_data, axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        standardized = self.raw_data / std
        cov = np.cov(standardized, rowvar=False)
        cov = np.atleast_2d(cov)
        return np.maximum(np.linalg.eigvalsh(cov), 0.0)

    @cached_property
    def _sqrt_cov_eigvals(self) -> np.ndarray:
        return np.sqrt(self._cov_eigvals)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------  

    @staticmethod
    def _max_eigengap(data: np.ndarray, k_nn: int, n_eig: int,
                      seed: int = 42) -> float:
        """Max consecutive eigengap of the normalized Laplacian from a k-NN graph."""
        nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='auto')
        nn.fit(data)
        dist, idx = nn.kneighbors(data)

        n = data.shape[0]
        sigma = np.maximum(dist[:, k_nn], 1e-10)

        src = np.repeat(np.arange(n), k_nn)
        dst = idx[:, 1:].ravel()
        d_sq = dist[:, 1:].ravel() ** 2
        scale = sigma[src] * sigma[dst]
        w = np.exp(np.maximum(-d_sq / scale, -500.0))

        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        vals = np.concatenate([w, w])

        W = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        degree = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-10)
        d_inv_sqrt = diags(1.0 / np.sqrt(degree))
        L_sym = speye(n) - d_inv_sqrt @ W @ d_inv_sqrt

        if n <= 30:
            eigvals = np.linalg.eigvalsh(L_sym.toarray())[:n_eig]
        else:
            # Seeded v0 for deterministic ARPACK convergence
            v0 = np.random.default_rng(seed).standard_normal(n)
            eigvals = eigsh(L_sym, k=n_eig, which='SM',
                            return_eigenvectors=False, v0=v0)
            eigvals = np.sort(np.real(eigvals))

        diffs = np.diff(eigvals[1:]) if len(eigvals) > 2 else np.diff(eigvals)
        return float(np.max(diffs)) if len(diffs) > 0 else 0.0

    # ------------------------------------------------------------------
    # Null tests — each returns a p-value (small p = structure detected)
    # ------------------------------------------------------------------

    @computed_field_cached_property()
    def gap_statistic(self) -> float:
        """Gap statistic p-value (Tibshirani, Walther & Hastie, 2001).

        Compares observed WCSS against a uniform null in the data's
        bounding box.  For each k >= 2, computes a z-score:

        ``z(k) = (E_null[log(WCSS)] - log(WCSS_obs)) / se_null``

        Returns ``1 - Φ(max_z)`` — the one-sided p-value of the most
        significant k.  Small p = strong evidence of structure.
        Returns NaN if data is unavailable.
        """
        if self.raw_data is None:
            return float('nan')

        wcss = self.wcss_by_k
        n, d = self.raw_data.shape
        # Exclude k >= n (perfect fit, WCSS = 0, undefined in log space)
        all_k_vals = sorted(k for k in wcss if 2 <= k < n) if wcss else []
        if not all_k_vals:
            return 1.0
        k_vals = self._sparse_k_vals(all_k_vals)
        n_ref = self._n_gap_references

        # Use per-sample mean squared error (WCSS/n) to account for
        # different sample counts between observed (outlier-removed) and
        # null (all n).  In log space: log(WCSS/n) = log(WCSS) - log(n).
        n_scored = self.n_scored_by_k
        log_wcss_obs = np.array([
            np.log(max(wcss[k], 1e-20)) - np.log(n_scored.get(k, n))
            for k in k_vals
        ])

        rng = np.random.default_rng(self.seed + 0)
        lo, _, spread = self._bbox

        # MiniBatchKMeans for n > 500 (3-5x faster; exact convergence
        # on uniform null data is unnecessary for a binary gate)
        _KM = MiniBatchKMeans if n > 500 else KMeans

        ref_data = np.empty((n, d), dtype=np.float64)
        log_wcss_refs = np.zeros((n_ref, len(k_vals)))
        for b in range(n_ref):
            rng.random(out=ref_data)
            ref_data *= spread
            ref_data += lo
            for j, k in enumerate(k_vals):
                km = _KM(n_clusters=k, n_init=1, max_iter=20,
                         random_state=self.seed + b)
                km.fit(ref_data)
                log_wcss_refs[b, j] = np.log(max(km.inertia_, 1e-20)) - np.log(n)

        mean_null = log_wcss_refs.mean(axis=0)
        std_null = log_wcss_refs.std(axis=0, ddof=1) * np.sqrt(1 + 1.0 / n_ref)
        gap = mean_null - log_wcss_obs

        z_per_k = np.where(std_null > self._eps, gap / std_null, 0.0)
        return float(stats.norm.sf(np.max(z_per_k)))

    _hopkins_max_d: int = 10

    @computed_field_cached_property()
    def hopkins_test(self) -> float:
        """Hopkins statistic p-value (Hopkins & Skellam, 1954).

        Under the null (spatially random), the Hopkins statistic
        ``H = sum(u) / (sum(u) + sum(w))`` follows ``Beta(m, m)``
        where m is the sample size.

        Returns the two-sided p-value: ``2 * min(P(H >= h), P(H <= h))``
        so that both clustering (H → 1) and regularity (H → 0) are
        detected.  Small p = strong evidence of spatial structure.
        Returns NaN if data is unavailable or n < 3.

        When ``d > _hopkins_max_d``, PCA reduces the data to at most
        ``_hopkins_max_d`` components (retaining ≥ 95% variance) before
        computing the statistic.  This avoids the distance-concentration
        problem where ``u**d`` and ``w**d`` converge for large d.
        """
        if self.raw_data is None:
            return float('nan')

        n, d = self.raw_data.shape
        data = self.raw_data

        if d > self._hopkins_max_d:
            from sklearn.decomposition import PCA

            max_components = min(n - 1, d)
            pca = PCA(n_components=max_components, random_state=self.seed)
            data = pca.fit_transform(data)
            # Keep only components needed for 95% variance, capped at max_d
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_keep = int(np.searchsorted(cumvar, 0.95)) + 1
            n_keep = min(max(n_keep, 2), self._hopkins_max_d)
            data = data[:, :n_keep]
            d = n_keep

        m = min(max(10, n // 5), n - 1)
        if m < 2 or n < 3:
            return float('nan')

        rng = np.random.default_rng(self.seed + 1)

        lo = data.min(axis=0)
        hi = data.max(axis=0)
        spread = np.where((hi - lo) < 1e-10, 1.0, hi - lo)
        random_points = (rng.random((m, d)) * spread + lo).astype(data.dtype)

        # Single NN tree with k=2 serves both queries
        nn = NearestNeighbors(n_neighbors=2, algorithm='auto')
        nn.fit(data)

        # Random → data: nearest neighbor (k=1)
        u_dist, _ = nn.kneighbors(random_points)
        u = u_dist[:, 0]  # closest neighbor

        # Data sample → data: true nearest (k=2, skip self at position 0)
        sample_idx = rng.choice(n, m, replace=False)
        w_dist, _ = nn.kneighbors(data[sample_idx])
        w = w_dist[:, 1]  # second closest = true nearest

        # Raise to power d so H follows Beta(m, m) under the uniform null.
        u_d = u ** d
        w_d = w ** d

        denom = u_d.sum() + w_d.sum()
        if denom < 1e-20:
            return float('nan')

        H = float(u_d.sum() / denom)

        p_lower = stats.beta.cdf(H, m, m)
        return float(2.0 * min(p_lower, 1.0 - p_lower))

    @cached_property
    def _standardized_data(self) -> np.ndarray:
        """Per-feature standardized raw data (zero mean, unit variance).

        Used by sigclust so that magnitude-dominated features don't hide
        shape-based cluster structure.
        """
        std = np.std(self.raw_data, axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        mean = np.mean(self.raw_data, axis=0)
        return (self.raw_data - mean) / std

    @computed_field_cached_property()
    def sigclust_test(self) -> float:
        """SigClust p-value (Liu et al., 2008).

        Tests whether the data is consistent with a single Gaussian.
        Operates on per-feature standardized data so that magnitude-
        dominated covariance doesn't hide shape-based cluster structure.

        Generates null samples from the best-fit Gaussian (matching the
        standardized data's eigenvalue spectrum), clusters each at k >= 2,
        and compares the observed cluster index ``CI = 1 - WCSS_k / WCSS_1``
        against the null distribution.  Both observed and null WCSS are
        computed in the same standardized space.

        Returns ``1 - Φ(max_z)`` — the one-sided p-value of the most
        significant k.  Small p = observed CI significantly exceeds
        the Gaussian null at some k.
        Returns NaN if data is unavailable, n < 4, or no valid k.
        """
        if self.raw_data is None:
            return float('nan')

        n, d = self.raw_data.shape
        if n < 4:
            return float('nan')

        # Work entirely in standardized space for consistent CI comparison.
        data_std = self._standardized_data
        _KM = MiniBatchKMeans if n > 500 else KMeans

        # Observed WCSS at a sparse subset of k values (binary gate
        # doesn't need every k; testing [2, 5, 10, 20] captures structure
        # at all scales).
        all_k_vals = sorted(k for k in self.wcss_by_k if 2 <= k < n)
        if not all_k_vals:
            return float('nan')
        k_vals = self._sparse_k_vals(all_k_vals)
        if not k_vals:
            return float('nan')

        obs_wcss_1 = max(float(np.var(data_std, axis=0).sum() * n), self._eps)
        obs_cis = np.empty(len(k_vals))
        for j, k in enumerate(k_vals):
            km = _KM(
                n_clusters=k, n_init=2, max_iter=50,
                random_state=self.seed,
            )
            km.fit(data_std)
            obs_cis[j] = 1.0 - km.inertia_ / obs_wcss_1

        # Null distribution from the correlation-matrix eigenvalue spectrum.
        n_ref = self._n_gap_references
        rng = np.random.default_rng(self.seed + 2)
        sqrt_eig = self._sqrt_cov_eigvals

        ref = np.empty((n, d), dtype=np.float64)
        null_cis = np.zeros((n_ref, len(k_vals)))
        for b in range(n_ref):
            rng.standard_normal(out=ref)
            ref *= sqrt_eig
            ref_wcss_1 = max(float(np.var(ref, axis=0).sum() * n), self._eps)
            for j, k in enumerate(k_vals):
                km = _KM(
                    n_clusters=k, n_init=1, max_iter=30,
                    random_state=self.seed + b,
                )
                km.fit(ref)
                null_cis[b, j] = 1.0 - km.inertia_ / ref_wcss_1

        mean_null = null_cis.mean(axis=0)
        std_null = null_cis.std(axis=0, ddof=1)

        # Effect-size gate: the raw CI excess must be practically
        # meaningful, not just statistically significant.  At large n,
        # z grows because std_null shrinks, but a CI excess below 0.05
        # (5% of total variance) is not meaningful structure — KMeans
        # can explain that much on any data by chance.
        effect = obs_cis - mean_null
        z_per_k = np.where(
            (std_null > self._eps) & (effect > 0.05),
            effect / std_null,
            0.0,
        )
        return float(stats.norm.sf(np.max(z_per_k)))

    @computed_field_cached_property()
    def spectral_gap_test(self) -> float:
        """Spectral gap p-value.

        Builds a k-NN graph, computes the normalized Laplacian's max
        consecutive eigengap, and compares against a Gaussian null
        (samples from N(mean, cov) matching the data's first two
        moments).

        A Gaussian null is used instead of column permutation because
        permutation preserves marginal distributions and fails to
        destroy axis-aligned cluster structure (common after PCA).
        The Gaussian null destroys all structure beyond the covariance.

        Returns ``1 - Φ(z)`` where z is the observed eigengap's z-score
        against the null.  Small p = eigengap significantly exceeds
        what's expected from a single Gaussian.
        Returns NaN if data is unavailable or n < 4.
        """
        if self.raw_data is None:
            return float('nan')

        n, d = self.raw_data.shape
        if n < 4:
            return float('nan')

        # Use standardized data so k-NN distances reflect shape
        # differences across all features, not just magnitude.
        data_std = self._standardized_data

        # Scale k_nn with n so sigma reflects local density, not the
        # dataset diameter.  At n=7, k_nn=2 (local); at n=1000+, k_nn=10.
        k_nn = max(2, min(10, int(np.sqrt(n))))
        n_eig = min(10, n - 1)
        obs_gap = self._max_eigengap(data_std, k_nn, n_eig, seed=self.seed)

        n_ref = self._n_gap_references
        rng = np.random.default_rng(self.seed + 3)
        sqrt_eig = self._sqrt_cov_eigvals

        null_gaps = np.empty(n_ref)
        for b in range(n_ref):
            ref = rng.normal(size=(n, d)) * sqrt_eig
            null_gaps[b] = self._max_eigengap(ref, k_nn, n_eig,
                                              seed=self.seed + 4 + b)

        mean_null = null_gaps.mean()
        std_null = null_gaps.std(ddof=1)
        if std_null < self._eps:
            return 0.0 if obs_gap > mean_null + self._eps else 1.0

        # Effect-size gate: require the eigengap excess to be at least
        # 15% larger than the null mean.  At large n, z grows because
        # std_null shrinks, but a marginal eigengap excess is not
        # meaningful cluster structure.
        effect = obs_gap - mean_null
        if mean_null > self._eps and effect / mean_null < 0.15:
            return 1.0

        z = effect / std_null
        return float(stats.norm.sf(z))

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------

    @computed_field_cached_property()
    def krzanowski_lai_index(self) -> dict[int, float]:
        """Krzanowski-Lai index (1988).

        Transforms the monotone WCSS curve into a peaked function via
        successive ratios of dimension-adjusted differences:
        ``DIFF(k) = (k−1)^(2/p)·W(k−1) − k^(2/p)·W(k)``
        ``KL(k) = |DIFF(k) / DIFF(k+1)|``

        Requires k−1 and k+1 to be present; boundary k values yield NaN.
        Natural direction: **maximize** — negated to minimize.
        """
        wcss = self.wcss_by_k
        p = self.n_features
        ks = self.k_values

        diff: dict[int, float] = {}
        for k in ks:
            if (k - 1) in wcss:
                diff[k] = (
                    (k - 1) ** (2.0 / p) * wcss[k - 1]
                    - k ** (2.0 / p) * wcss[k]
                )

        result: dict[int, float] = {}
        for k in ks:
            if k in diff and (k + 1) in diff and abs(diff[k + 1]) > self._eps:
                val = abs(diff[k] / diff[k + 1])
                val *= -1
                result[k] = val
            else:
                result[k] = np.nan
        return result

    @computed_field_cached_property()
    def hartigan_index(self) -> dict[int, float]:
        """Hartigan index (1975).

        WCSS improvement from k to k+1, corrected for sample size:
        ``H(k) = (W(k)/W(k+1) − 1) · (n − k − 1)``

        Traditional rule: choose the smallest k where H(k) ≤ 10.
        Requires k+1 to be present; k_max yields NaN.
        Natural direction: **minimize**.
        """
        wcss = self.wcss_by_k
        n = self.n_samples
        ks = self.k_values

        result: dict[int, float] = {}
        for k in ks:
            if (k + 1) in wcss and wcss[k + 1] > self._eps:
                val = (wcss[k] / wcss[k + 1] - 1.0) * (n - k - 1)
                result[k] = val
            else:
                result[k] = np.nan
        return result

    @computed_field_cached_property()
    def distortion_jump_index(self) -> dict[int, float]:
        """Distortion jump index (Sugar & James, 2003).

        Power transformation of WCSS amplifies the elbow signal:
        ``J(k) = W(k)^{−p/2} − W(k−1)^{−p/2}``

        When W(k) collapses to zero (perfect clusters) the jump is +∞;
        when both W(k) and W(k−1) are zero the jump is 0.
        Requires k−1 to be present; k_min yields NaN.
        Natural direction: **maximize** — negated to minimize.
        """
        wcss = self.wcss_by_k
        p = self.n_features
        ks = self.k_values
        exp = -p / 2.0

        result: dict[int, float] = {}
        for k in ks:
            if (k - 1) in wcss:
                w_k = wcss[k]
                w_km1 = wcss[k - 1]
                if w_k <= self._eps and w_km1 <= self._eps:
                    # Both k and k-1 already perfect — no further jump.
                    val = 0.0
                elif w_k <= self._eps:
                    # WCSS collapsed to zero at k: maximally large jump.
                    val = np.inf
                else:
                    # Normal case (WCSS is monotone non-increasing, so
                    # w_km1 >= w_k; W^{-p/2} is monotone decreasing in W).
                    d_k = w_k ** exp
                    d_km1 = w_km1 ** exp if w_km1 > self._eps else np.inf
                    val = d_k - d_km1
                val *= -1
                result[k] = val
            else:
                result[k] = np.nan
        return result

    @computed_field_cached_property()
    def log_wcss_acceleration_index(self) -> dict[int, float]:
        """Log-WCSS acceleration index — second difference of log WCSS.

        ``D2(k) = log W(k−1) − 2·log W(k) + log W(k+1)``

        Peaks at the sharpest elbow in the log-WCSS curve. Numerically
        stable via the log transformation. Requires k−1 and k+1 to be
        present; boundary k values yield NaN.
        Natural direction: **maximize** — negated to minimize.
        """
        wcss = self.wcss_by_k
        ks = self.k_values

        def _log(w: float) -> float:
            return np.log(w) if w > self._eps else np.log(self._eps)

        result: dict[int, float] = {}
        for k in ks:
            if (k - 1) in wcss and (k + 1) in wcss:
                val = _log(wcss[k - 1]) - 2.0 * _log(wcss[k]) + _log(wcss[k + 1])
                val *= -1
                result[k] = val
            else:
                result[k] = np.nan
        return result


    @computed_field_cached_property()
    def xu_index(self) -> dict[int, float]:
        """Xu index (Xu, 2019).

        Information-theoretic penalty balancing fit (WCSS) against
        model complexity (k):
        ``Xu(k) = p · log₂(√(WCSS / (n · p²))) + log₂(k)``

        Natural direction: **minimize**.
        """
        wcss = self.wcss_by_k
        n = self.n_samples
        p = self.n_features
        ks = self.k_values
        log2 = np.log2

        result: dict[int, float] = {}
        for k in ks:
            w = wcss[k]
            if w < self._eps:
                w = self._eps
            val = p * log2(np.sqrt(w / (n * p * p))) + log2(k)
            result[k] = val
        return result


def has_cluster_structure(ckm: CrossKMetrics, alpha: float = 0.05) -> bool:
    """Combined null test gate — grouped agreement.

    Tests are grouped by methodology:

    - **Distance-based** (gap statistic, Hopkins): detect spatial
      concentration.  Share a blind spot on non-uniform unimodal
      distributions (Gaussians, exponentials, heavy-tailed).
    - **Model-based** (SigClust, spectral gap): detect deviation
      from a single-component model.  Robust to non-uniformity
      but less sensitive to moderate overlap.

    Structure is declared only when **both groups** have at least
    one individually significant test (p < *alpha*).  This eliminates
    false positives from the shared distance-based blind spot while
    preserving sensitivity to genuine clusters.

    Falls back to 2-of-N majority if one group is entirely
    unavailable.  Returns True (permissive) if fewer than 2 tests
    are computable.
    """

    dist_pvals = [p for p in (ckm.gap_statistic, ckm.hopkins_test)
                  if not np.isnan(p)]
    model_pvals = [p for p in (ckm.sigclust_test, ckm.spectral_gap_test)
                   if not np.isnan(p)]

    if not dist_pvals or not model_pvals:
        all_pvals = dist_pvals + model_pvals
        if len(all_pvals) < 2:
            return True
        return sum(1 for p in all_pvals if p < alpha) >= 2

    dist_sig = any(p < alpha for p in dist_pvals)
    model_sig = any(p < alpha for p in model_pvals)
    return dist_sig and model_sig

