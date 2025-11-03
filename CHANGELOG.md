Changelog
=========

Development
-----------

* Switch build to use pyproject.toml and uv 
* Change pyproject.toml build system to hatchling 
* Update to example data to include GHI
* Update `load_test_data` function to always pull from GitHub.
* Add comparison groups. This feature is still in development. Final API is unfinished.
* Consolidate clustering for hourly model and CG clustering

opendsm-1.2.6
-----------

* Fixed bug in `from_series` instantiation of daily data class

opendsm-1.2.5
-----------

* Expose SpectralClustering's assign_labels options. `discretize` and `cluster_qr` are not always deterministic with seed
* Added more metrics to BaselineMetrics

opendsm-1.2.4
-----

* Bug fix of metrics that squeaked through

opendsm-1.2.3
-----

* Spectral clustering is now seeded appropriately
* Fixed bug making seed in main hourly settings not be passed to clustering settings
* Including new metrics in baseline_metrics

opendsm-1.2.2
-----

* Change daily model to use CVRMSE_adj and PNRMSE_adj as intended
* Autocorrelation function is now consistent

opendsm-1.2.1
-----

* Revert how autocorr is calculated prior to 1.2.0

opendsm-1.2.0
-----

* Add hourly model uncertainty
* Daily model uses BaselineMetrics natively now, accessible from model.baseline_metrics
* Data classes now accept dictionaries to modify DQ criteria. This is an R&D feature
* Migrate to modern Python logger interface to solve deprecation warnings

opendsm-1.1.0
-----

* Updated the Hourly model
* Performed new optimization for Hourly model configuration
* Developed adaptive robust weighting per hour-of-day for the hourly model
* Updated adaptive loss function. Previously it assumed too large of a range of outliers and made choosing alphas < 0 unlikely
* Altered clustering methodology, it now uses spectral clustering
* Changed temperature binning to be fixed bins
* Made temporal bins/temperature bins act together on temperature
* Disallow negative CVRMSE in Hourly model
* Added daily CVRMSE >= 0 and PNRMSE sufficiency requirements
* Partially updated Daily model to use baseline_metrics
* Changed extreme values warning flag to check using IQR rule instead of median +- IQR which is incorrect
* Fix warning data on `high_frequency_temperature_data` warning.
* Squash numpy divide-by-zero warnings in caltrack Hourly metrics.

opendsm-1.0.0
-----

* Initial OpenDSM release

eemeter-4.1.1
-----

* Add GHI sufficiency check requiring 90% coverage for each month
* Add weights propogation from data class to daily model via "weights" column
* Converted daily model settings from attrs to pydantic
* Refactored daily model initial guess optimization to use consolidated optimize function
* Add experimental daily weighting for hourly model fitting (if one day is crazy, it will be down weighted in the fit)

eemeter-4.1.0
-----

* Add new hourly model to support solar meters and improve nonsolar results

eemeter-4.0.8
-----

* Add github action to publish to pypi
* Bump to latest packages and remove all deprecation/future warnings as of 2024-12-20.
* Allow identical observations to not raise exception for daily model in `linear_fit`.
* Handle ambiguous and nonexistent local times when creating billing dataclass 
* Fix serialization and deserialization of hourly CalTRACK metrics.
* Rename HourlyBaselineData.sufficiency_warnings -> HourlyBaselineData.warnings
* Add disqualification field to HourlyBaselineData and HourlyReportingData
* Fix bug where HourlyBaselineData and HourlyReportingData wasn't actually NaNning zero rows when `is_electricity=True`.
* Constrain eemeter daily model balance points to T_min_seg and T_max_seg rather than T_min and T_max.
* Fix bug in `linear_fit` due to SciPy's `theilslopes(y, x)` not following the same order as `linregress(x, y)`

eemeter-4.0.7
-----

* Handle ambiguous and nonexistent local times when creating daily dataclass

eemeter-4.0.6
-----

* Update docs.
* Update typehints on core daily and utility functions.
* Minor change to loading test data to ensure the reporting period is a year ahead of the baseline period.

eemeter-4.0.5
-----

* Flip slope when deserializing legacy hdd_only models

eemeter-4.0.4
-----

* Add support for deserializing legacy hourly models
* Fix legacy daily model deserialization

eemeter-4.0.3
-----

* Move masking behavior for rows with missing temperature from reporting dataclass to prediction output
* Add disqualification check to billing model predict()

eemeter-4.0.2
-----

* Force index to use nanosecond precision
* Compute coverage using same offset as initial reads to fix issues when downsampling hourly data
* Update test data location
* Fix bug in daily plotting to remove NaN values if input
* Refactor sufficiency criteria to be more explicit and easier to manage

eemeter-4.0.1
-----

* Correct dataframe input behavior and final row temperature aggregation
* Remove unnecessary datetime normalization in order to respect hour of day
* Convert timestamps in certain warnings to strings to allow serialization
* Allow configuration of segment_type in HourlyModel wrapper


eemeter-4.0.0
-----

* Update daily model methods, API, and serialization
* Provide new API for hourly model to match daily syntax and prepare for future additions
* Add baseline and reporting dataclasses to support compliant initialization of meter and temperature data

eemeter-3.2.0
-----

* Addition of modules and amendments in support of international facility for EEMeter, including principally:
* Addition of quickstart.py; updating setup.py and __init__/py accordingly.
* Inclusion of temperature conversion amendments to design_matrices; features; and derivatives.
* Addition of new tests and samples.
* Amendments to tutorial.ipynb.
* Addition of eemeter international.ipynb.
* Change .iteritems() to .items() in accordance with pandas>=2.0.0
* .get_loc(x, method=...) to .get_indexer([x],method=...)[0] in accordance with pandas>=2.0.0
* Updated mean() to mean(numeric_only=True) in accordance to pandas>=2.0.0
* Updated tests to work with pandas>=2.0.0
* Update python version in Dockerfile.
* Update other dependencies (including adding rust) in Dockerfile.
* Remove pinned dependencies in Pipfile.
* Relock Pipfile (and do so inside of the docker image).
* Update pytests to account for changes in newer pandas where categorical variables are no longer included in `df.sum().sum()`.
* Clarify the functioning of start, end and max_days parameters to `get_reporting_data()` and `get_baseline_data()`.

eemeter-3.1.1
-----

* Update observed_mean calculation to account for solar (negative usage) to provide
sensible cvrmse calculations.

eemeter-3.1.0
-----

* Remove missing hour_of_week categories in the CalTrack hourly methods so they predict null for those hours. 

eemeter-3.0.0
-----

* Remove python27 support.
* Update Pipfile lock.
* Update `fit_temperature_bins` to potentially take an `occupancy_lookup` in order to
  fit different temperature bins for occupied/unoccupied modes. *This changes the args passed to eemeter.create_caltrack_hourly_segmented_design_matrices, where it now requires a set of bins for occupied and unoccupied temperatures separately.*
* Update CalTRACK hourly model formula to use different bins for occupied and
  unoccupied mode.

eemeter-2.10.11
-------

* Fix tests and make changes to ensure tests pass on pandas version 1.2.1.
* Fix bug in segmentation.py causing a section of tutorial to fail.

eemeter-2.10.0
------

* Add additional terms into ModelMetrics() class which can be used in fractional savings uncertainy computations.

eemeter-2.9.2
-----

* Remove fixing of versions of libraries in setup.py to avoid unforeseen issues with library updates.

eemeter-2.9.1
-----

* Fix versions of libraries in setup.py to avoid unforeseen issues with library updates.

eemeter-2.9.0
-----

* Clarify blackout period.

eemeter-2.8.6
-----

* Fix issue with `get_reporting_data` and `get_baseline_data` when passing data with non-UTC timezones.

eemeter-2.8.5
-----

* Add functions to clean billing/daily data according to caltrack rules.

eemeter-2.8.4
-----

* Further limit segments used in hourly `totals_metrics` to only calculate when weight=1.

eemeter-2.8.3
-----

* Update hourly `totals_metrics` calculation to properly use only the segment of the model.

eemeter-2.8.2
-----

* Add `totals_metrics` to hourly models.

eemeter-2.8.1
-----

* Fix bug with `get_baseline_data` in regards to recent addition of `n_days_billing_period_overshoot` kwarg.

eemeter-2.8.0
-----

* Update `get_baseline_data` to allow for limit to billing overshoot using `n_days_billing_period_overshoot` kwarg.

eemeter-2.7.7
-----

* Add function to clean billing data to fit caltrack specifications (`clean_caltrack_billing_data`).

eemeter-2.7.6
-----

* Update io functions to support latest pandas (>=0.24.x).
* Update documentation for CalTRACK Hourly methods.
* Add tutorial.

eemeter-2.7.5
-----

* Fix completeness check for `get_terms` for last term.

eemeter-2.7.4
-----

* Make more usable outputs for the `get_terms` function (list of eemeter.Term objects).

eemeter-2.7.3
-----

* Update `as_freq` so it has an optional `include_coverage` parameter where it returns a dataframe with one column including the percent coverage of data used to create each sample.

eemeter-2.7.2
-----

* Fixes the columns that are given in an empty prediction result called with the
  ` with_design_matrix=True` flag set for caltrack usage per day methods.
* Update bug report github issue template.
* Add test for `as_freq`.

eemeter-2.7.1
-----

* Change `as_freq` to handle all Null series.

eemeter-2.7.0
-----

* Add `get_terms` method to allow splitting reporting data into any number
  of terms specified by day length.

eemeter-2.6.0
-----

* Change `fit_caltrack_hourly_model` so it returns a `CalTRACKHourlyModelResults` object rather than a `CalTRACKHourlyModel`, in order to bring it in line with the `caltrack_usage_per_day` model outputs.

eemeter-2.5.4-post1
-----------

* Update MANIFEST.in to fix release and update `./bump_version.sh` script
  to remove build directories.

eemeter-2.5.4
-----

* Add data fields to the `DataSufficiency` even if there are no warnings when calculating sufficiency.

eemeter-2.5.3-post2
-----------

* Attempt 2 to fix release .whl file by removing local build and dist
  directories before running `python setup.py upload`.

eemeter-2.5.3-post1
-----------

* Fix release .whl file which had some extra directories.
* Add draft MAINTAINERS.md.

eemeter-2.5.3
-----

* Fix `metered_savings` behavior so that it does not fail to compute error bands when there is 0 variance in the baseline.

eemeter-2.5.2
-----

* Fix `as_freq` behavior to preserve sum and add a null last index at the target
  frequency if necessary.

eemeter-2.5.1
-----

* Capture an additional exception type (`KeyError`) in recently adjusted
  `get_baseline_data` and `get_reporting_data` methods.

eemeter-2.5.0
-----

* Add parameters to `get_baseline_data` and `get_reporting_data` to help make
  these methods a bit more correct for billing data.
* Preserve nulls properly in `as_freq`.
* Update jupyter version to be compatible with latest tornado version.

eemeter-2.4.0
-----

* Fix for bug that occasionally leads to `LinAlgError: SVD did not converge` error when fitting caltrack hourly models by addressing multi-collinearity when only a single occupancy mode is detected

eemeter-2.3.1
-----

* Hot fix for bug that occasionally leads to `LinAlgError: SVD did not converge` error when fitting caltrack hourly models by converting the weights from `np.float64` ton `np.float32`.

eemeter-2.3.0
-----

* Fix bug where the model prediction includes features in the last row that should be null.
* Fix in `transform.get_baseline_data` and `transform.get_reporting_data` to enable pulling a full year of data even with irregular billing periods

eemeter-2.2.10
------

* Added option in `transform.as_freq` to handle instantaneous data such as temperature and other weather variables.

eemeter-2.2.9
-----

* Predict with empty formula now returns NaNs.

eemeter-2.2.8
-----

* Update `compute_occupancy_feature` so it can handle instances where there are less than 168 values in the data.

eemeter-2.2.7
-----

* SegmentModel becomes CalTRACKSegmentModel, which includes a hard-coded check that the same hours of week are in the model fit parameters and the prediction design matrix.

eemeter-2.2.6
-----

* Reverts small data bug fix.

eemeter-2.2.5
-----

* Fix bug with small data (1<week) for hourly occupancy feature calculation.
* Bump dev eeweather version.
* Add `bump_version` script.
* Filter two specific warnings when running tests:
  statsmodels pandas .ix warning, and eemeter model fitting warning.

eemeter-2.2.4
-----

* Add `json()` serialization for `SegmentModel` and `SegmentedModel`.

eemeter-2.2.3
-----

* Change `max_value` to float so that it can be json serialized even if the input is int64s.

eemeter-2.2.2
-----

* Add warning to `caltrack_sufficiency_criteria` regarding extreme values.

eemeter-2.2.1
-----

* Fix bug in fractional savings uncertainty calculations using billing data.

eemeter-2.2.0
-----

* Add fractional savings uncertainty to modeled savings derivatives.

eemeter-2.1.8
-----

* Update so that models built with empty temperature data won't result in error.

eemeter-2.1.7
-----

* Update so that models built from a single record won't result in error.

eemeter-2.1.6
-----

* Update multiple places where `df.empty` is used and replaced with `df.dropna().empty`.
* Update documentation for running CalTRACK hourly methods.

eemeter-2.1.5
-----

* Fix zero division error in metrics calculation for several metrics that
  would otherwise cause division by zero errors in fsu_error_band calculation.

eemeter-2.1.4
-----

* Fix zero division error in metrics calculation for series of length 1.

eemeter-2.1.3
-----

* Fix bug related to caltrack billing design matrix creation during empty temperature traces.

eemeter-2.1.2
-----

* Add automatic t-stat computation for metered savings error bands, the
  implementation of which requires expicitly adding scipy to setup.py
  requirements.
* Don't compute error bands if reporting period data is empty for metered
  savings.

eemeter-2.1.1
-----

* Fix degree day ranges (30-90) for prefab caltrack design matrix creation
  methods.
* Fix the warning for total degree days to use total degree days instead of
  average degree days.

eemeter-2.1.0
-----

* Update the `use_billing_presets` option in `fit_caltrack_usage_per_day_model`
  to use a minimum data sufficiency requirement for qualifying CandidateModels
  (similar to daily methods).
* Add an error when attempting to use billing presets without passing a weights
  column to facilitate weighted least squares.

eemeter-2.0.5
-----

* Give better error for duplicated meter index in compute temperature features.

eemeter-2.0.4
-----

* Change metrics input length error to warning.

eemeter-2.0.3
-----

* Apply black code style for easy opinionated PEP 008 formatting
* Apply JSON-safe float conversion to all metrics.

eemeter-2.0.2
-----

* Cont. fixing JSON representation of NaN values

eemeter-2.0.1
-----

* Fixed JSON representation of model classes

eemeter-2.0.0
-----

* Initial release of 2.x.x series
