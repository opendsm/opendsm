# Comparison Groups Methodology

A comparison group corrects a treatment meter's counterfactual for the exogenous
conditions the counterfactual model could not anticipate. This page describes how
the populations are structured, how a comparison group is selected, how the
correction is computed, how uncertainty is propagated, and how meters are
disqualified along the way.

## Populations and granularity

A population is a set of per-meter eemeter models with their baseline data and,
once available, their reporting data. A treatment group is uniform. Every meter
shares one granularity, one timezone, and one fuel type, and solar and non-solar
meters are never mixed in the same population. Uniformity is required because the
correction combines meters into a single treatment-scale series, and combining
across granularities or timezones would misalign the timesteps being summed.

The comparison pool may be finer than the treatment but never coarser. Fineness
is ordered billing, then daily, then hourly, and the pool must be at least as
fine as the treatment. An hourly pool can support a daily treatment because finer
data can always be aggregated up to a coarser cadence, whereas the reverse would
require inventing sub-period structure the coarse data does not carry. When the
pool is finer, it is aggregated to the treatment's cadence before the correction
runs.

Loadshapes for selection are always built at the treatment population's
granularity. The default time period is chosen per treatment granularity and
then passed to both populations, so a finer pool aggregates to the treatment's
cadence rather than producing a loadshape of a different length. Mismatched
loadshape lengths would otherwise make the two populations incomparable.

### The billing substrate and read periods

A billing meter is handled differently because its reads do not arrive on a
regular calendar. Internally a billing meter rides a contiguous daily substrate,
and the raw read datetimes are consumed during data processing rather than kept
on the processed frame. The correction therefore recovers the read periods that
partition that substrate, aggregates the daily substrate and the comparison-pool
matrices to those periods, and only then runs the difference-in-differences.
Correcting at the meter's own read cadence keeps the correction on the same
footing as the reads the utility actually billed.

Read periods are inferred from the daily observed spread. Within a single read
the per-minute spread rate is constant, so day-length-normalized observed values
are equal within a read and jump between reads. Boundaries are drawn at those
jumps, between finite days only, because a missing day belongs to no read. Three
adjustments make the inference robust. A spurious short segment, such as a
re-estimated day, is absorbed into its nearest-rate neighbor. A span of
near-identical consecutive reads that never jumps is subdivided into equal
periods at the detected median cadence. A caller may also pass explicit read
boundaries, which override inference outright.

A read period is credible only within a valid length range, currently 25 to 70
days. A meter whose cadence cannot be recovered, because it has no observed data,
a constant spread with no resolvable cadence, or an inferred period outside the
valid range, is recorded on the disqualification ledger and dropped rather than
corrected on a guess. When a prior correction is supplied for an extension, its
comparison group is reused as it stands and its confirmed read periods are
frozen with only the trailing period re-inferred, so extending the reporting
window can move neither the group nor an earlier boundary.

## Selection

Selection chooses which pool meters form the comparison group for each treatment
meter. Four methods are available, and each is a single call that returns a
normalized selection the correction can consume. Each method has its own complete
description linked below.

- **CG clustering** groups treatment and pool meters together by loadshape and
  assigns each treatment meter the clusters it falls in. It is the only method
  that gives every treatment meter its own comparison group, and it matches on
  the model-error basis so the comparison group's residuals track the
  treatment's. See [CG clustering](docs/cg_clustering.md).
- **Individual meter matching** pairs each treatment meter with its closest pool
  meters by a distance on their loadshapes. The union of all matches is treated
  by the correction as one shared comparison group. See
  [individual meter matching](docs/individual_meter_matching.md).
- **Stratified sampling** draws a pool sample balanced against the treatment
  distribution over supplied stratification features, binning both populations
  and sampling the pool bin by bin. See
  [stratified sampling](docs/stratified_sampling.md).
- **Random sampling** draws a pool sample uniformly at random without reference
  to the treatment, useful as a baseline or on a very homogeneous pool. See
  [random sampling](docs/random_sampling.md).

The loadshape basis differs by method. CG clustering matches on modeling error,
the residual between each meter's model and its observed baseline usage, because
the correction operates on the comparison group's own model-versus-observed
residual. The matching and sampling methods match on the modeled load, because
they aim to reproduce the treatment's load level.

Selection depends only on baseline data. This is what lets selection run once at
the start of a program and be reused as reporting data accumulates.

## Correction

The correction is a generalized difference-in-differences applied per treatment
meter, per comparison-group meter, at each timestep. The corrected counterfactual
is $m_{cT} = m_T - s_{CG}(m_{CG} - o_{CG})$, where $m_T$ is the treatment model,
$m_{CG} - o_{CG}$ is the comparison meter's own reporting-period model error, and
$s_{CG}$ is a scale factor. Subtracting the scaled comparison error from the
treatment model removes the shared exogenous movement from the counterfactual. As
a worked example, a comparison meter whose reporting model predicted 100 units
against 72 observed carries a 28-unit error that, scaled to the treatment meter,
lowers its counterfactual.

Three scale forms are available, ordinary, percent, and absolute percent, each
resting on a different assumption about how the exogenous effect relates the two
meters. Within a cluster the per-meter corrections are combined by a
model-magnitude weighted average with the concentration capped, the clusters are
combined by their treatment weights, and the correction degrades per cluster so a
sparse cluster does not abort a timestep. A treatment meter with fewer than five
comparison-group meters is excluded and recorded on the ledger. The full
derivation, the scale assumptions, the aggregation and weight-cap mathematics,
the degradation semantics, and the uncertainty propagation are in
[the correction](docs/correction.md).

## Uncertainty

The uncertainty outputs are honest heuristic bands, not calibrated intervals, and
they should be read as such. Three separate constructions combine to produce
them, and each carries an approximation.

The per-timestep model uncertainty is a t-scaled prediction-interval band in the
ASHRAE style, propagated from the eemeter model fit. It is treated as sigma-like
when combined, but it is not a calibrated one-sigma quantity. The significance
level is set by `alpha`, currently 0.10.

Treatment observed uncertainty is a separate, optional input set on the treatment
population. It never enters the correction math; instead it is threaded through
the correction onto the corrected series and the savings step combines it into
the savings band in quadrature. An explicit value passed to the savings
computation overrides the threaded one.

Combining these bands over time is exact only at hourly cadence. At hourly
cadence the per-timestep band already reconstructs the ASHRAE hourly aggregate,
so summing it in quadrature over a period reconstructs that aggregate exactly. At
daily and billing cadence the per-timestep band is itself only a prediction
interval for that single period, so a period's uncertainty is a
quadrature-summed prediction-interval band rather than an ASHRAE aggregate share.
Unifying these two constructions is future work.

Taken together, these outputs quantify uncertainty that would otherwise be
unquantified, but they are not yet calibrated to a stated coverage level.
Calibration is an open area, and readers should treat the bands as relative
indicators rather than exact confidence statements.

## Data sufficiency and the disqualification ledger

Every meter dropped anywhere in the stream is recorded on a shared ledger, with
columns for the meter id, the pipeline stage that dropped it, the origin of the
problem, a short human reason, and a detail string carrying the verbatim eemeter
warning or the computed statistic that triggered the drop. The ledger exists so
that no meter disappears silently. A savings total can always be traced back to
the meters behind it and the meters excluded from it.

Sufficiency is checked at two stages against a group window. A group window is the
union of the attached meters' spans, or an explicit override. At selection, each
treatment meter and each selected pool meter must carry finite observed and
temperature data over at least a minimum fraction of the baseline group window,
currently 0.9. At correction, the same coverage floor is applied over the
reporting group window. A meter below the floor is pruned from the selection or
correction, not merely noted, because the correction skips only absent meters and
a note alone would leave a thin meter corrected.

The reporting data carries one additional enforcement. A meter whose reporting
observed data trips an eemeter observed or joint disqualification, such as
repeated identical non-zero reads or too many co-missing days, is dropped before
prediction. That is a structural data-quality failure rather than a signal.

No extreme-value screen is applied to the reporting-period observed usage itself.
This is deliberate. An intervention or an exogenous shock can legitimately drive
reporting usage to an extreme, and that extreme is exactly the quantity the
analysis measures. Screening reporting magnitude would discard the signal along
with any noise. The magnitude controls that do exist act on the correction rather
than the data. A correction cap bounds each correction to a multiple of the
treatment model magnitude, defaulting to three times, and for solar meters the
cap applies only where the comparison model magnitude falls below a small
threshold.
