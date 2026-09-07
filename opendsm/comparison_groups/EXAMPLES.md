# Comparison Groups Examples

These walkthroughs run against the ComStock daily test data bundled with
OpenDSM. Each builds a treatment group of eight meters and a comparison pool of
sixty, then measures savings with a comparison-group correction. The first
example runs the stream for one meter. The second stages the work so selection
happens once and correction runs later as reporting data arrives.

## Building the populations

A population is built from a mapping of meter id to a fitted eemeter model and
its data. The helper below fits one daily model per meter and returns that
mapping. The same helper serves both the treatment group and the pool.

```python
from opendsm.common.test_data import load_test_data
from opendsm.comparison_groups import (
    ComparisonGroupSelection,
    ComparisonPool,
    MeterAnalysis,
    TreatmentGroup,
    compute_savings,
    correct_reporting,
    select_comparison_group,
)
from opendsm.eemeter.models import DailyModel


def build_meters(df_b, df_r, ids):
    meters = {}

    for mid in ids:
        baseline_df = df_b.xs(mid, level="id").reset_index()
        meters[str(mid)] = {
            "model": DailyModel().fit(baseline_df, is_electricity_data=True),
            "baseline_df": baseline_df,
            "reporting_df": df_r.xs(mid, level="id").reset_index(),
        }

    return meters


df_b, df_r = load_test_data("daily_treatment_data")
ids = sorted(df_b.index.get_level_values("id").unique())
treatment_ids = ids[:8]
pool_ids = ids[8:68]

treatment = TreatmentGroup.from_fit_models(build_meters(df_b, df_r, treatment_ids))
pool = ComparisonPool.from_fit_models(build_meters(df_b, df_r, pool_ids))
```

`from_fit_models` infers the granularity from the model instances and validates
that every meter shares one timezone and fuel type. The populations now hold
baseline and reporting data, so the whole stream can run at once.

## Example 1: one meter, straight through

Selection runs once for the whole treatment population, because the sampling
methods choose a group by comparing treatment meters against the pool as a set.

```python
selection = select_comparison_group(treatment, pool, method="cg_clustering")

print(selection.method.value)   # "cg_clustering"
print(selection.basis)          # "error"
print(selection.clusters.shape) # (60, 3)
```

Everything downstream is per meter. `MeterAnalysis` takes that selection and one
treatment id; `run` corrects the meter's reporting model against its comparison
group and computes savings.

```python
analysis = MeterAnalysis(selection, treatment, pool, str(treatment_ids[0]))
analysis.run(aggregation="monthly")
```

The savings frame reports observed and corrected energy, the savings between
them, an uncertainty band, a percent savings, and the coverage fraction behind
the period.

```python
analysis.savings_result.savings.head()
#        id   period       observed  ...  savings_unc  pct_savings  coverage
# 0  108618  2019-01  113307.769409  ...     13243.71     0.160235       1.0
```

A meter that cannot be corrected at all raises `MeterCorrectionError`, which
carries the ledger rows explaining why, so a loop can record the failure and
continue. Individual timesteps that cannot be corrected come back as NaN rows
instead, with the period's `coverage` below one.

The meter log reports what was dropped for this meter and for the pool meters in
its comparison group, across every stage run so far. On this clean data nothing
is dropped, so the log is empty, but it always carries the same columns.

```python
log = analysis.meter_log()
print(list(log.columns))  # ['id', 'stage', 'origin', 'reason', 'detail']
print(len(log))           # 0
```

Any calendar aggregation is available by rerunning `savings`. A total collapses
the whole reporting year to one row for the meter.

```python
analysis.savings(aggregation="total")
analysis.savings_result.savings
```

## Example 2: staged across the reporting year

Selection depends only on baseline data, so it can run before any reporting data
exists. Populations are built with baseline data alone, and the selection is
serialized to JSON for reuse.

```python
def build_baseline_meters(df_b, ids):
    meters = {}

    for mid in ids:
        baseline_df = df_b.xs(mid, level="id").reset_index()
        meters[str(mid)] = {
            "model": DailyModel().fit(baseline_df, is_electricity_data=True),
            "baseline_df": baseline_df,
        }

    return meters


treatment = TreatmentGroup.from_fit_models(build_baseline_meters(df_b, treatment_ids))
pool = ComparisonPool.from_fit_models(build_baseline_meters(df_b, pool_ids))

selection = select_comparison_group(treatment, pool, method="cg_clustering")
selection_json = selection.to_json()
```

The JSON can be stored and reloaded later. Deserialization recomputes the
selection fingerprint and raises if the stored tables are inconsistent, so a
loaded selection is guaranteed to match the one that was saved. The fitted
models persist alongside it through `TreatmentGroup.to_json` and
`ComparisonPool.to_ndjson`, which the caller reassembles on the other side.

```python
selection = ComparisonGroupSelection.from_json(selection_json)
```

When reporting data arrives, it is attached to the populations by id. The caller
passes cumulative data, so each attachment replaces the previous reporting slice
rather than appending to it.

```python
reporting_t = {str(mid): df_r.xs(mid, level="id").reset_index() for mid in treatment_ids}
reporting_p = {str(mid): df_r.xs(mid, level="id").reset_index() for mid in pool_ids}

treatment.add_reporting_data(reporting_t)
pool.add_reporting_data(reporting_p)
```

The correction and savings then run against the attached data, one meter at a
time. Because each correction recomputes over the full cumulative window,
rerunning it as more reporting data accumulates keeps the earlier corrected
values stable while the uncertainty reflects the longer period.

```python
meter_id = str(treatment_ids[0])
correction = correct_reporting(selection, treatment, pool, meter_id)
print(correction.granularity)  # "daily"

savings = compute_savings(correction, aggregation="annual")
savings.savings
```

Passing a prior correction to `correct_reporting` reuses the prior's comparison
group as it stands, so a pool meter whose reporting lags the treatment's can
neither enter nor leave the group, and checks the new result against the prior
where the periods overlap, so an incremental run cannot silently move values that
were already reported. The prior must belong to the same meter and the same
selection.
