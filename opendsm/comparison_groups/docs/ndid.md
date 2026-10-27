# The NDID correction

Normalized difference-in-differences (NDID) is a fourth correction algorithm,
selected with `CGCorrectionSettings(algorithm="normalized_difference_in_differences",
ndid=NDIDSettings(...))`. It answers the same question as the existing
algorithms, how much of a comparison group's reporting-period model error to
transfer onto a treatment meter, but it never divides by an instantaneous load.

The absolute percent form scales each comparison meter's error by the ratio of
the treatment's prediction to that meter's prediction. When a comparison meter's
prediction approaches zero the ratio diverges, and its uncertainty term diverges
faster. Net-metered solar at peak production is the common case: the net
prediction is small by cancellation while the gross loads and their errors are
large. The existing guards (an exact-zero exception, a correction cap, magnitude
weights, a weight cap) bound the symptom rather than remove the division.

NDID normalizes each meter's error by that meter's typical load for that kind of
hour, learned from its own baseline, and combines comparison meters at each
timestep with weights built from how closely each meter's predicted operating
state matches the treatment's. It is bounded by construction, needs no cap on the
correction, uses no meter's error history as a weight, and makes no assumption
about when loads or production peak. It runs at hourly, daily and billing
cadence on any window length, and for the same admitted pool a row has the same
value whether the reporting window was delivered at once or in pieces.

## Definitions

Here $t$ indexes reporting timesteps, $i$ the admitted pool meters, and $k$ the
comparison-group clusters. Each timestep falls in a cell $q(t)$ of a calendar
grid described below. For the treatment, $m_T(t)$ is the model prediction. For
pool meter $i$, $m_i(t)$ and $o_i(t)$ are its prediction and observed usage.
$s_i(t)$ and $s_T(t)$ are the typical loads for the timestep's cell, read from
each meter's baseline profile, and $S_i$, $S_T$ are their annual scales. At
billing cadence the typical load is a per-day value multiplied by the read
period's day count.

The normalized residual and the predicted relative load (the state) are

$$
e_i(t) = \frac{m_i(t) - o_i(t)}{s_i(t)}, \qquad
x_i(t) = \frac{m_i(t)}{s_i(t)}, \qquad
x_T(t) = \frac{m_T(t)}{s_T(t)}.
$$

A pool meter is available at $t$ when its prediction and observed value are
finite. Its typical load is always positive, because a meter with no baseline
load is not admitted (see below).

## The baseline profile

Every fitted model carries a baseline profile, computed from its baseline
predictions and residuals with interpolated rows dropped as the fit drops them.
The profile is computed at fit time and serialized beside the baseline metrics.
A model serialized before the profile existed loads without one, and the
population computes it on demand from the meter's attached baseline data the
first time it is needed. A meter with neither raises an error naming it.

The cells are hour of week by month at hourly cadence (2016 cells), day of week
by month at daily cadence (84 cells), and month at billing cadence (12 cells). A
billing row is one read period, indexed by its midpoint, with observed and
predicted values per day. Per cell the profile accumulates the row count $N_q$,
the mean squared prediction and the mean squared residual. Each mean square is
smoothed across cells by a count-weighted kernel average,

$$
\bar v_q = \frac{\sum_{q'} G(q, q') N_{q'} v_{q'}}{\sum_{q'} G(q, q') N_{q'}},
\qquad G = \prod \frac{1}{1 + u^2},
$$

where each Cauchy factor measures a circular distance between cell positions:
hour of day, weekday, and day of year, each divided by a bandwidth (2 hours, 1
day and 28 days by default, set at fit time through `ProfileSettings`). A factor
is dropped when the cadence step is not shorter than its period. Cauchy factors
never underflow, so every cell receives a value whenever any row exists, and a
short baseline still fills all twelve months through the calendar factor.

The profile stores, per cell, the typical load $\sqrt{\overline{P^2}_q}$, the
residual RMS $\sqrt{\overline{R^2}_q}$, and the state spread, the smoothed RMS of
each row's prediction relative to its cell's typical load, minus one. It also
stores the autocorrelation of the normalized residual series at lags 1 to 168
(hourly) or 1 to 7 (daily), none at billing. Three scalars complete it: the
residual coefficient of variation, the annual scale $S = \sqrt{\overline{m^2}}$, and a robust
Yeo-Johnson exponent $\lambda$ fitted to the normalized residuals after
median and MAD standardization ($\lambda = 1$ when the MAD is zero).

A model whose kept baseline predictions are all zero, or that has no kept rows,
stores a zero profile. The correction does not admit such a pool meter, records
it on the ledger as `no_baseline_load`, and raises if the treatment is one.

## Pooled laws

At each correction run the admitted pool meters' profiles are pooled once, and
nothing is re-estimated from reporting data. With $r_i[q] = \text{rms}_i[q]^2 /
s_i[q]^2$ the per-cell relative residual variance, the population law is the mean
of $r_i$ over meters, and each cluster's law shrinks its own mean toward it,

$$
v_k[q] = \frac{M_k \, \overline{r}_k[q] + D \, v_{pop}[q]}{M_k + D},
$$

with $M_k$ the cluster's meter count and $D$ the `cluster_pseudo_count`. The
cluster's state spread is blended the same way on the squared scale. The
cluster's Yeo-Johnson exponent $\lambda_k$ is the median of its meters'
exponents, its autocorrelation the mean of theirs, and $S_{ref}$ the median
annual scale. A size-law exponent $\gamma$ is taken from the settings, or from
the sector when unset (0 for commercial, 0.25 for residential). The slope of log
residual coefficient of variation against log annual scale is reported as a
diagnostic only. Clusters are the distinct non-negative labels among the
admitted meters. A selected cluster left without an admitted meter is dropped
and its weight renormalized over the rest.

## Weights

Within cluster $k$ at timestep $t$, each available meter's weight is

$$
w_i(t) = C\!\left(\frac{x_i - x_T}{h_k}\right)
\, C\!\left(\frac{\ln S_i - \ln S_T}{b_S}\right)
\left(\frac{S_i}{S_{ref}}\right)^{2\gamma}
\, \text{trust}_i(t),
\qquad C(u) = \frac{1}{1 + u^2},
$$

where $h_k = \kappa \cdot \text{spread}_k[q(t)]$ with $\kappa$ the
`state_bandwidth_factor`, and $b_S$ the `size_bandwidth` (the size factor is 1
when it is unset). The state factor matches meters whose predicted operating
state resembles the treatment's counterfactual state. The Cauchy kernel never
reaches zero, so as close matches thin the estimate leans on the nearest meters
and reaches the cluster's plain mean only in the limit.

The trust factor is a one-pass robustness weight. The residuals are standardized
by their weighted median and MAD, passed through the Yeo-Johnson transform at
the cluster's fixed $\lambda_k$ so that skew does not decide what counts as
extreme, standardized again, and each meter receives $\min(1, 3 / |z_i|)$. A
meter beyond three robust spreads of the cross-section is discounted, while one whose
value is merely large is counted nearly in full.

## Point and band per cluster

$$
\hat e_k(t) = \frac{\sum_i w_i e_i}{\sum_i w_i}, \qquad
n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},
$$

$$
\text{se}^2 = \frac{\sum_i w_i^2 (e_i - \hat e_k)^2}{\left(\sum_i w_i\right)^2}
\cdot \frac{n_{\text{eff}}}{\max(n_{\text{eff}} - 1, 10^{-9})},
\qquad
\text{var}_k(t) = \beta \, \text{se}^2 + (1 - \beta) \frac{v_k[q(t)]}{n_{\text{eff}}},
$$

with $\beta = \text{clip}(n_{\text{eff}} - 1, 0, 1)$. The variance is a one-sigma
sampling variance of $\hat e_k$: purely cross-sectional once $n_{\text{eff}} \ge
2$ and purely the pooled baseline law at $n_{\text{eff}} = 1$. No interval
factor is applied inside the correction, so `alpha` does not enter NDID.

## Heterogeneity per cluster

The band above says how well the cluster's mean is known. The treatment also
sits some distance from that mean, by an offset of its own that no number of
pool meters removes, and the cross-section measures how large such offsets are.
With $p_i = w^0_i / \sum_j w^0_j$ the weights before the trust factor,
$\tilde p_i = w_i / \sum_j w_j$ the weights after it, and
$n_0 = 1 / \sum_i p_i^2$,

$$
D_k(t) = \sum_i p_i \left(e_i - \hat e_k\right)^2, \qquad
g_i = p_i - 2 p_i \tilde p_i + \tilde p_i^2, \qquad
c_k(t) = \sum_i g_i,
$$

$$
h_k(t) = \beta_0 \max\!\left(0, \frac{D_k - \sum_i g_i \, r_i[q(t)]}{c_k}\right)
+ (1 - \beta_0) \, v_k[q(t)], \qquad \beta_0 = \text{clip}(n_0 - 1, 0, 1).
$$

The spread is taken over the weights before trust because the treatment is not
trimmed: a pool meter that trust discounts is still evidence of how far a meter
can sit from the mean. It is centered on the trusted mean because that is the
estimate the treatment is corrected with. For independent meters under fixed
weights, meter $i$'s variance enters the expected spread with weight $g_i$, so
subtracting each meter's own baseline noise $r_i$ at that weight and dividing by
$c_k$ leaves the heterogeneity whatever the meters' noise. The cluster's pooled
law would not serve here, because the size law and the state kernel favor
meters whose noise can differ from the cluster's average. With equal weights
that trust leaves whole, $c_k = 1 - 1/n_0$. The treatment's own noise is
already in its band. Below $n_0 = 2$ the estimate blends toward the pooled law,
which stands in for the heterogeneity a single meter cannot identify, so the
band does not narrow for lack of meters.

## Combination and outputs

With $T_k(t)$ the treatment's cluster weights renormalized over the clusters
that contribute at $t$,

$$
m_{cT}(t) = m_T(t) - s_T(t) \sum_k T_k(t) \, \hat e_k(t), \qquad
\sigma^2_{CG}(t) = s_T(t)^2 \sum_k T_k(t)^2 \, \text{var}_k(t),
$$

$$
\sigma^2_H(t) = s_T(t)^2 \sum_k T_k(t) \, h_k(t), \qquad
\sigma_{cT}(t) = \sqrt{\sigma_T(t)^2 + \sigma^2_{CG}(t) + \sigma^2_H(t)},
$$

where $\sigma_T$ is the treatment model's own band. The cluster weights enter
$\sigma^2_{CG}$ squared, because the clusters' sampling errors are independent
and average down, and $\sigma^2_H$ unsquared, because the treatment's own
offset does not shrink when its weight spreads over several clusters. The
spread of the cluster means between clusters, which a treatment drawn from one
cluster would add, is not included. The correction frame gains `cg_var`
($\sigma^2_{CG}$), `cg_heterogeneity_var` ($\sigma^2_H$) and `cg_effective_n`
($\sum_k T_k n_{\text{eff},k}$),
and the result carries `cg_acf`, the treatment-weighted mean of the clusters'
autocorrelations, and `diagnostics` with the pooled-law summary. A row whose
treatment prediction is not finite, or at which no cluster contributes, is NaN.
A row with no typical load takes no correction, and both variances are 0 there.

## Aggregation

`compute_savings` sums a period's two comparison-group variances with their
lags rather than in quadrature. With $\sigma_t = \sqrt{\sigma^2_{CG}(t)}$ and
$\eta_t = \sqrt{\sigma^2_H(t)}$ over the period's rows in time order (zero where
NaN) and $\rho_j$ the result's `cg_acf`,

$$
\text{var}_{CG} = \sum_t \sigma_t^2
+ 2 \sum_{j=1}^{K} \rho_j \sum_{t=1}^{n-j} \sigma_t \sigma_{t+j},
\qquad
\text{var}_{H} = \sum_t \eta_t^2
+ 2 \sum_{j=1}^{K} \rho_j \sum_{t=1}^{n-j} \eta_t \eta_{t+j},
$$

and the period variance is $\sum_t (\sigma_{cT,t}^2 - \sigma^2_{CG,t} -
\sigma^2_{H,t} + \sigma_{O,t}^2) + \text{var}_{CG} + \text{var}_{H}$, with
$\sigma_O$ the observed uncertainty. The two series are independent of each
other, so no product of one with the other enters. A series' negative lag sum
is set to zero, so that series falls back to quadrature. With every
$\rho_j = 0$ this equals the existing path exactly, and at billing cadence the
read periods are summed in quadrature. Error shared by several treatments
corrected against one pool is not combined across treatments, because the
correction runs one meter at a time. A combination across treatments would treat
$\sigma_{CG}$ as shared among the treatments corrected against one pool, in full
only where their weights on the pool coincide, and $\sigma_H$ as independent,
which is why the two are reported separately.

## Settings

`NDIDSettings` holds `state_bandwidth_factor` (default 1.0), `size_bandwidth`
(off by default, natural-log units), `gamma` (in [0, 0.5], unset by default),
`sector` (`"commercial"` or `"residential"`, required when `gamma` is unset) and
`cluster_pseudo_count` (default 20). With NDID selected, the legacy fields
`weight_cluster_aggregation`, `weight_cap`, `outlier_rejection` and
`correction_cap` are rejected if set. `min_window_coverage` applies as for the
other algorithms.

At timesteps where a pool meter shares the treatment's relative state, NDID's
transfer equals the absolute percent form's, so with $\gamma = 0.5$ NDID
reproduces that method up to the state kernel and the trust step, and with
$\gamma = 0$ it is the plain mean of normalized residuals.

## Approximations and limits

- The residual autocorrelation is computed over the baseline series with
  dropped rows closed up, so a lag spanning a gap is counted as if the rows
  were adjacent.
- The profile's $\lambda$ is fitted to residuals standardized by median and MAD,
  and the class then applies its own Huber standardization, close to the
  identity on an already standardized series. The correction standardizes each
  cross-section by weighted median and MAD before applying the fixed $\lambda$.
  The residual mismatch between the two standardizations is not corrected.
- The treatment's own band is a t-scaled prediction interval, and it is combined
  with a one-sigma comparison-group term, the same heuristic mix the existing
  algorithms use.
- A row's correction, `cg_var` and `cg_heterogeneity_var` depend only on its own
  reporting cross-section
  and frozen baseline quantities, but the admitted pool can differ between
  deliveries through `min_window_coverage`, which is evaluated per delivered
  window. At hourly cadence the model's own band (`modeled_unc`) is an aggregate
  over the delivered window, so `corrected_unc` also moves with the window.
  Those are the two sources of difference between progressive and one-shot
  delivery.
- The baseline noise the heterogeneity subtracts comes from in-sample baseline
  residuals, which understate reporting-period noise and so leave the
  heterogeneity high, and from residuals that contain misfit the cluster
  shares, which the correction removes, and so leave it low. Their net is not
  measured.
- The heterogeneity at a row rests on that row's cross-section alone, so it is
  as noisy as the cluster is small. Its degrees of freedom are about
  $(n_0 - 1) \left(h_k / (h_k + \bar r_k)\right)^2$, with $\bar r_k$ the
  subtracted noise, falling toward a few where the noise is as large as the
  heterogeneity, so an interval for a single row needs a multiplier on those
  degrees of freedom.
- The truncation at zero biases the heterogeneity upward where it is small
  against the noise and the cluster is small: on synthetic cross-sections with
  equal weights and no true heterogeneity, the mean estimate is about a quarter
  of the noise variance at five meters and an eighth at twenty. Sums over many
  rows inherit that bias.
- The heterogeneity's lag terms multiply square roots of noisy row estimates,
  which understates them: with estimates independent across rows, by up to a
  quarter at five to ten meters. Offsets that persist across rows make
  neighbouring estimates agree and narrow the gap. A treatment's offset may
  also persist beyond the longest autocorrelation lag, so aggregates over longer
  periods understate it, as they do the correction's own error.
- Contamination of the pool and the pool's representativeness for the treatment
  have no statistic. The in-sample autocorrelation may understate aggregate
  bands. In a first billing window with few matches, the band rests on the pooled
  baseline term. A minority of pool meters whose change exceeds three robust
  spreads of the cross-section is partly discounted.
