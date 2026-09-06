# Comparison Groups

A savings measurement compares a building's metered energy against a
counterfactual, the energy the building would have used without an intervention.
The counterfactual comes from an eemeter model fit on a baseline year and
projected into the reporting year. That projection carries whatever the model
could not see. Weather beyond the model's response, a shifting economy, and
broad shocks such as a pandemic all move reporting-year usage without any
intervention, and a plain model reads those movements as savings.

The comparison groups module removes that exogenous movement. A set of
non-participant meters, chosen to resemble the treatment meters, experiences the
same exogenous conditions but received no intervention. The gap between what the
comparison group's models predicted and what it actually used estimates the
exogenous effect, and subtracting that gap from each treatment meter's
counterfactual leaves a savings measurement with the shared movement removed.

## The pipeline

The module is a stream of stages, each consuming the previous stage's output.

- **Populations** (`TreatmentGroup`, `ComparisonPool`) hold per-meter fitted
  eemeter models and their data. A treatment group shares one granularity,
  timezone, and fuel type. A pool may be finer than the treatment and is
  aggregated up when needed.
- **Selection** (`select_comparison_group`) chooses the comparison group. Four
  methods are available (CG clustering, individual meter matching, stratified
  sampling, random sampling), each a one-line call that returns a
  `ComparisonGroupSelection`.
- **Correction** (`correct_reporting`) applies the difference-in-differences for
  one treatment meter at its received cadence and returns a `CorrectionResult`
  of corrected reporting-period series.
- **Savings** (`compute_savings`) reduces that correction to avoided energy,
  summed at any calendar aggregation, and returns a `SavingsResult`.

Selection runs once for the whole treatment population, because the sampling
methods need the group to choose from. Everything downstream is per meter:
`MeterAnalysis` takes a selection and one treatment id, and its `correct()` and
`savings()` methods run the two stages in order, with `run()` chaining both.

## Two ways to use it

The module supports two patterns because a reporting year usually arrives in
pieces rather than all at once.

The first is straight-through. Populations are built with baseline and reporting
data attached, and one `run()` call produces a savings result. This suits a
retrospective analysis where the full reporting period is already in hand.

The second stages the work across the reporting year. Selection depends only on
baseline data, so it can run once at the start, be serialized, and be reused as
reporting data accumulates. Each later correction recomputes over the full
cumulative window, which keeps past corrected values stable while the
uncertainty reflects everything seen so far, and passing the previous
correction as a prior freezes the meter's comparison group, so a pool meter
whose data lags cannot change the group under values already reported. This
suits an ongoing program that reports incrementally.

## Documentation

- [Methodology](METHODOLOGY.md) explains the granularity rules, the selection
  methods, the correction mathematics, the uncertainty framing, and the
  disqualification ledger.
- [Examples](EXAMPLES.md) walks through runnable code for both usage patterns.
