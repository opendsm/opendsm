gridmeter / eesampling — archived changelog
===========================================

Archived release history of the comparison-group methods as developed by Recurve in
the standalone `gridmeter` package (originally `eesampling`), before they were merged
into opendsm. `gridmeter` is a trademark of Recurve; the code was donated to opendsm,
the trademark was not. This file is frozen and will not be updated. Versions 0.0.1
through 0.5.5 predate the first public release (0.6.0) and were never published. The
final entry `gridmeter-2.0.0a1` is a pre-release alpha (a 2.0 rewrite) that was never
published to PyPI; it is recorded here as the last standalone state of the package before
the methods were merged into `opendsm.comparison_groups`.

gridmeter-2.0.0a1 (2024-06-20)
------------------------------

* Restructured the package into method-scoped subpackages (`clustering`, `individual_meter_matching`, `random_sampling`, `stratified_sampling`, `_utils`) with a unified top-level API that now exports `Data`, `Data_Settings`, `Clustering`/`Clustering_Settings`, `IMM`/`IMM_Settings`, `Random_Sampling`/`RS_Settings`, `Stratified_Sampling`/`SS_Settings`/`DSS_Settings`, and `load_tutorial_data`.
* Added a new `Data` input class that accepts either loadshapes (stacked or unstacked) or raw time series and converts them to a canonical loadshape, with configurable aggregation (`MEAN`/`MEDIAN`), loadshape type (observed/modeled/error), and time period (hourly, day-of-week, weekday/weekend, month, and seasonal variants).
* `Data` performs granularity checks against the requested time period, linear interpolation of missing values, datetime/timezone coercion (to UTC), deterministic duplicate resolution (sort by absolute value, keep first), a minimum-data-percent inclusion threshold, and exposes `loadshape`, `features`, `ids`, and `excluded_ids`; all comparison-group methods now consume `Data` objects rather than raw frames.
* Added pydantic-based settings classes (`Data_Settings`, clustering `Settings`, IMM `Settings`, random-sampling `Settings`) with field validation, replacing the prior keyword-argument configuration; settings are passed at construction and methods take treatment/pool `Data`.
* Added loadshape clustering (`Clustering`): bisecting k-means over an fPCA projection (`FPCA_MIN_VARIANCE_RATIO`), configurable normalization (`MIN_MAX` or none) before clustering, lower/upper bounds on bisection count that scale with data size, min-cluster-size outlier handling, variance-ratio/distance scoring to pick the cluster composition, treatment-to-cluster matching with selectable loss (`MAE`/`SSE`/`adaptive` or float alpha), seeded reproducibility, optional multiprocessing, and cluster/loadshape plotting.
* Reworked individual meter matching: added `SELECTION_METHOD` of `legacy`, `minimize_meter_distance`, or `minimize_loadshape_distance` (the loadshape method solving a HiGHS-based weighted fit), `DISTANCE_METRIC` choice (euclidean/seuclidean/manhattan/cosine), `N_MATCHES_PER_TREATMENT`, chunked treatment processing (`N_TREATMENTS_PER_CHUNK`), an optional `MAX_DISTANCE_THRESHOLD` post-filter, and an `ALLOW_DUPLICATE_MATCHES` toggle with deterministic duplicate-distance resolution.
* Added difference-in-differences computation (`_utils/diff_of_diff.py`): a percent correction factor from comparison-group observed/counterfactual and a corrected treatment counterfactual.
* Added a random-sampling comparison-group method (`Random_Sampling`) with seeded selection by total meter count or per-treatment count (mutually exclusive).
* Migrated stratified sampling into the new settings-based API (`Stratified_Sampling` with `SS_Settings`/`DSS_Settings`) consuming `Data`, alongside the retained model/bin-selection/equivalence/diagnostics machinery.
* Bundled NREL-Comstock/Resstock-derived example datasets (features and seasonal/day-of-week/hourly loadshape CSVs plus hourly parquet) surfaced through `load_tutorial_data`.

gridmeter-1.1.0 (2021-06-29)
----------------------------

* Add `DistanceMatching` for selecting comparison groups by usage-pattern distance: computes `cdist` between treatment and pool meters, supports per-feature `weights` and chunking via `n_treatments_per_chunk` for memory-bounded matching of large treatment sets.
* Make duplicate-match resolution deterministic: when multiple treatment meters map to the same comparison meter, keep the closest by distance and re-match the rest against remaining pool meters, replacing the prior arbitrary/random assignment (`match py closest, not by random`).
* Add `param_selection` module to rank candidate matching parameters by KL divergence between treatment and comparison distributions (`get_kl_divs`) and select a decorrelated subset via a correlation threshold (`choose_params`/`get_params`).
* Fix equivalence-feature averaging in `StratifiedSamplingBinSelector`: the treatment groupby was missing its `feature_index` key, so per-feature means were computed incorrectly.
* Add usage tutorial/documentation and expand test coverage (multi-round matching, distance-calc selection).

gridmeter-1.0.1 (2020-11-19)
----------------------------

* Packaging-only follow-up to the 1.0.0 `GRIDmeter` rename; no functional, API, or behavior changes.
* Fixed the package metadata: corrected `__title__` to `"GRIDmeter"` and removed the leftover `"!! No longer maintained -- please use gridmeter instead !!"` placeholder from `__description__`, restoring it to `"Tools for stratified sampling for comparison groups"`.
* Bumped `__version__` to `1.0.1`.

gridmeter-1.0.0 (2020-11-19)
----------------------------

* First release under the GRIDmeter name; the package, module path, and imports are renamed from `eesampling` to `gridmeter` (`from gridmeter import ...`). No functional or API behavior changes accompany the rename.
* Purely a rename-and-repackage release: source/test directories, `setup.py`, Docker/docs config, and the tutorial notebook move from `eesampling/` to `gridmeter/`. All comparison-group functionality (stratified sampling, `StratifiedSamplingBinSelector`, equivalence checks, diagnostics, synthetic data) is carried over unchanged from `eesampling-0.10.x`.

eesampling-0.10.1 (2020-11-19)
------------------------------

* Maintenance-only release: no code, API, or behavior changes over `0.10.0`.
* Marks `eesampling` as deprecated -- the package `__description__` now reads "No longer maintained -- please use `gridmeter` instead", directing users to the renamed `gridmeter` package.

eesampling-0.10.0 (2020-11-19)
------------------------------

* Final release under the `eesampling` name; the package was renamed `gridmeter`.
* Importing `eesampling` now emits a `DeprecationWarning` (and prints a notice) directing users to install `gridmeter` instead. No API or functional changes otherwise.
* Packaging maintenance: bumped version `0.9.1` -> `0.10.0`, added `twine` to requirements and a `bump_version.sh` script.

eesampling-0.9.1 (2020-11-14)
-----------------------------

* `results_as_json()` now reports comparison-pool equivalence distances alongside the treatment and selected-sample values, under `chisquare_averages.comparison_pool`, so the full pool can be compared against the chosen sample.
* Otherwise a maintenance release: version bump, `.pypirc` packaging cleanup, and tutorial-notebook updates; no other behavior or API changes.

eesampling-0.9.0 (2020-10-09)
-----------------------------

* Rewrote the equivalence calculation used for comparison-group bin selection, decoupling the per-quantile mean computation from the distance reduction so equivalence is computed once and passed in rather than recomputed during fitting (much faster).
* Changed the `Equivalence` API: replaced `n_bins` with `n_quantiles` plus a `how` distance selector (`euclidean` or `chisquare`); `compute()` now returns reshaped per-feature/per-quantile output DataFrames (`equiv_x`, `equiv_y`) alongside the scalar `distance`.
* Changed the equivalence input format for bin selection from a long-form DataFrame (`df_for_equivalence` with `equivalence_groupby_col`/`equivalence_value_col`/`equivalence_id_col`) to a wide feature matrix plus an id array (`equivalence_feature_matrix`, `equivalence_feature_ids`), with `equivalence_method` (default `chisquare`) and `equivalence_quantile_size`.
* Added an `ids_to_index` helper mapping meter IDs to row indices in the feature matrix.
* Removed the dead `Diagnostics`/`records_based_equivalence` code path that the bin selector previously used for equivalence.

eesampling-0.8.0 (2020-09-24)
-----------------------------

* Add synthetic data generation (`SyntheticMeter`, `SyntheticPopulation`, `SyntheticTreatmentPoolPopulation`) for testing and tutorials, with synthetic treatment drawn as a subset of the pool and a cache key for reuse.
* Expose all public classes at the top level: `StratifiedSampling`, `StratifiedSamplingBinSelector`, `StratifiedSamplingDiagnostics`, `ModelSamplingException`, and the synthetic-data classes.
* Make `n_samples_approx` an upper bound: with `relax_n_samples_approx_constraint=True`, the minimum sampled:treatment ratio may be violated to reach the requested count when bins are short on pool data; centralized in `get_counts_and_update_n_samples_approx`.
* Add a separate `id_col` for the core dataframe, distinct from `equiv_id_col`, so identity and equivalence keys can differ.
* Speed up the equivalence computation in diagnostics and fix ID-column handling; fix a diagnostics plotting bug.
* Add a tutorial Jupyter notebook (with Docker/Jupyter setup) and an Apache-2.0 `LICENSE`.

eesampling-0.7.0 (2020-09-16)
-----------------------------

* Renamed the train/test vocabulary throughout the public API to treatment/pool: `df_train`/`df_test` parameters become `df_treatment`/`df_pool`, `min_n_train_per_bin` becomes `min_n_treatment_per_bin`, and `min_n_sampled_to_n_train_ratio` becomes `min_n_sampled_to_n_treatment_ratio`.
* `DiagnosticsRunner` constructor labels changed: `train_label`/`test_label` are now `treatment_label`/`pool_label` (defaulting to `"treatment"`/`"pool"`), and the corresponding diagnostic accessors are renamed (e.g. `n_sampled_to_n_treatment_ratio`).
* Breaking change with no compatibility shims: callers passing the old `df_train`/`df_test` keyword arguments must update to the new names.

eesampling-0.6.1 (2020-09-16)
-----------------------------

* Packaging only: correct the project `__url__` in `__version__.py` to the `recurve-methods/comparison-groups/eesampling` GitHub path; no functional changes over `0.6.0`.

eesampling-0.6.0 (2020-09-16)
-----------------------------

* First public release of `eesampling`, the comparison-group sampling library (later renamed GRIDmeter).
* Replaces the initial random-sampling prototype (`sampling_model.py`) with a stratified-sampling pipeline: `StratifiedSampling` (`model.py`), `StratifiedSamplingBinSelector` (`bin_selection.py`), and binning primitives `Binning`/`Bin`/`MultiBin`/`BinnedData` plus `sample_bins` (`bins.py`).
* Adds `Diagnostics`/`DiagnosticPlotter` (`diagnostics.py`) with `t_and_ks_test` and balance plots for assessing how well the sampled comparison group matches the treatment group.
* Public API exports `StratifiedSampling` and the `ModelSamplingException` error from the package root.
* `StratifiedSamplingBinSelector` targets `n_samples_approx=5000` comparison meters with a `min_n_sampled_to_n_train_ratio=0.25` floor; `relax_n_samples_approx_constraint` controls whether `n_samples_approx` is treated as a hard target or an upper bound.

eesampling-0.5.5
----------------

* Update results serialization.

eesampling-0.5.4
----------------

* Add kwargs and results serialization.

eesampling-0.5.3
----------------

* Separate bin selection into a different class.

eesampling-0.5.2
----------------

* Fix issue with naming during equivalence chisquare checking of diagnostics (this needs to be refactored later).

eesampling-0.5.1
----------------

* Renamed `min_bin_size` to `min_n_train_per_bin`.
* Move BinnedData to bins.py.
* Added chisquared equivalence option.
* Add equivalence via a separate dataframe.

eesampling-0.5.0
----------------

* Added some unit tests for modelling and some test framework.
* Generalized Diagnostics so that .plot_equivalence(...) can also plot the comparison pool.
* Changed automatic n_samples_approx to use the maximum number of samples available (based on how many test values are in the "worst" bin) rather than use binary search.
* Renamed n_outputs to n_samples_approx

eesampling-0.4.2
----------------

* Fix random seed so that numpy random seeding for pertubation is happening in the right place.
* Make a copy of the dataframe in the `_perturb()` function.

eesampling-0.4.1
----------------

* Add random seed option.

eesampling-0.4.0
----------------

* Support fixed-width or variable-with bins
* Auto-choose number of outputs via binary search

eesampling-0.3.3
----------------

* Scatter plot has fixed y scales and correct size


eesampling-0.3.2
----------------

* Fix bug if not using auto-bin

eesampling-0.3.1
----------------

* Remove plotly dependency

eesampling-0.3.0
----------------

* Simplify plotting
* Add auto_bin option


eesampling-0.2.0
----------------

* Big refactor, add plotting diagnostics.
* Add plotly support

eesampling-0.1.0
----------------

* Initial create of model.

eesampling-0.0.1
----------------

* Initial creation of library.
