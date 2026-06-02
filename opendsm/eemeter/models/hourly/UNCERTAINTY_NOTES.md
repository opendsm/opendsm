# Hourly uncertainty redesign — status & open items

**Status: PARKED, not validated. Do not merge without resolving the open items below.**

This branch builds on the glaring-bug-fix branch
`feature/uncertainty-fixes-and-cleanup` (t_stat dof, edf λ₂, `.equals` crash fix,
legacy-path isolation). Rebase on that; the bugs are not duplicated here.

## What this branch changes

- **Per-point band** → per-hour-of-day empirical `(1 − uncertainty_alpha)`
  quantile of `|residual|` (`HourlyBaselineMetrics.hour_uncertainty`):
  distribution-free, heteroscedastic; replaces the Gaussian `total/√n` form.
- **`BaselineMetrics.residual_vif`** = `1 + 2 Σ_{k=1..48} ρ_k` from the
  chronological residual autocorrelation; the per-point band is scaled by
  `√residual_vif` so a quadrature (`sum_quad`) aggregate is not too narrow.
- `predicted_unc = uncertainty_scale_factor · √residual_vif · hour_uncertainty[hour]`.
- Removed the now-unused `ReportingMetrics.predicted_data_point_unc`.
- Design contract: per-point bands must be valid when summed in quadrature.

Coverage on the test meter (ComStock 116756, in-sample, nominal 90%): per-point
99.8% (conservative — `√VIF` inflates the per-hour quantile), daily-quadrature
92%. (Original was 67% / 20.5%.)

## Open items — BLOCKING

1. **Not validated ≥ ASHRAE 14.** Multi-meter check (5 ComStock meters): ours
   beats ASHRAE AR(1) at daily/weekly — ASHRAE `(1+ρ₁)/(1−ρ₁)` blows up (38–48)
   for high-`ρ₁` meters because it extrapolates `ρ₁` geometrically, wrong for a
   diurnal ACF — but monthly is ambiguous. No out-of-sample / placebo validation.

2. **`residual_vif` is window-dependent; a single fixed scalar is the wrong
   object.** The empirical variance inflation of the residual SUM grows with the
   aggregation window (5-meter mean: daily ≈ 14, monthly ≈ 25; it does NOT
   saturate). The fixed `L=48` scalar (≈ 14) overshoots daily/weekly and
   undershoots monthly — the billing horizon. The accurate object is the
   window-aware exact finite-window VIF `1 + 2 Σ_{k<W}(1 − k/W) ρ_k`, which tracks
   the empirical inflation at every window but cannot be precomputed as one
   per-point scalar → it breaks the quadrature contract. **DECISION NEEDED:**
   fixed scalar (simple, daily-tuned) vs window-aware (accurate, relaxes quadrature).

3. **Relocate `HourlyBaselineMetrics`** out of `common/metrics.py` into this
   directory (`eemeter/models/hourly/`): it is model-specific and the shared
   `BaselineMetrics` must stay generic. `residual_vif` stays on the generic base.

## Out-of-sample notes

- Analytical in-sample optimism is small: `κ ≈ √((N+p)/(N−p)) ≈ 1.08`
  (N ≈ 365 fit-days, p ≈ 28 nonzero coef/hour) — swamped by the `√VIF` margin.
- Contiguous seasonal holdouts drop coverage sharply, but that is seasonal
  extrapolation (predicting a season absent from training), not in-sample
  optimism; production fits a baseline year and predicts the same seasons.

## Related, but NOT this branch

- Billing monthly aggregate ≈ 58% undercoverage is the same RSS-independence
  problem one level up — cross-**day** correlation among days in a billing period
  (`DailyModel`/`BillingModel` `sum_quad`), not cross-hour. Separate fix; billing
  still uses ASHRAE.
