In this example, we'll walk through creating a Daily Model and predicting usage with it.

If you'd like to follow along, be sure to begin with the following imports. Numpy is used for one example of a disqualified meter, and matplotlib is used for plotting the data. Neither are required imports if you're not following along with the example precisely.

```python
import numpy as np
import matplotlib.pyplot as plt

import eemeter
```

### Loading Example Data
With our imports set, we can begin by loading some example data. Here we use a built in utility function to load some prepared test data.

This function returns two dataframes of daily electricity data, one for the baseline period and one for the reporting period.

```python
df_baseline, df_reporting =  eemeter.load_test_data("daily_treatment_data")
```

If we inspect these dataframes, we will notice that there are 100 meters for you to experiment with, indexed by meter id and datetime.

```python
print(df_baseline)
```

??? Returns
    ```
    id	    datetime		            temperature	observed
    108618	2018-01-01 00:00:00-06:00	-2.384038	16635.193673
            2018-01-02 00:00:00-06:00	1.730000	15594.051162
            2018-01-03 00:00:00-06:00	13.087946	11928.025899
            2018-01-04 00:00:00-06:00	4.743269	14399.333812
            2018-01-05 00:00:00-06:00	4.130577	14315.101721
    ...	...	...	...
    120841	2018-12-27 00:00:00-06:00	52.010625	1153.749811
            2018-12-28 00:00:00-06:00	35.270000	1704.076968
            2018-12-29 00:00:00-06:00	29.630000	2151.225729
            2018-12-30 00:00:00-06:00	34.250000	1331.123954
            2018-12-31 00:00:00-06:00	43.311250	1723.397349
    ```

To simplify things, we will filter down to a single meter for the rest of the example. Let's filter down to id 108618.

```python
df_baseline_108618 = df_baseline.loc[108618]
df_reporting_108618 = df_reporting.loc[108618]
```

If we inspect one of these dataframes, we will now notice that only a single meter is present with 365 days of data in each dataframe. 

```python
print(df_baseline_108618)
```

??? Returns
    ```
    datetime		            temperature	observed
    2018-01-01 00:00:00-06:00	-2.384038	16635.193673
    2018-01-02 00:00:00-06:00	1.730000	15594.051162
    2018-01-03 00:00:00-06:00	13.087946	11928.025899
    2018-01-04 00:00:00-06:00	4.743269	14399.333812
    2018-01-05 00:00:00-06:00	4.130577	14315.101721
    ...	...	...
    2018-12-27 00:00:00-06:00	46.602066	4528.347029
    2018-12-28 00:00:00-06:00	38.346724	5647.646228
    2018-12-29 00:00:00-06:00	28.614456	5338.377496
    2018-12-30 00:00:00-06:00	29.186923	6280.238343
    2018-12-31 00:00:00-06:00	36.510441	4966.566443
    ```

Also notice the general structure of these dataframes for a single meter. We have three columns:
1. A timezone-aware datetime index.
2. A temperature column (float) in Fahrenheit (be sure to convert any other units to Fahrenheit first).
3. Observed meter usage (float). Our example is electricity data in kWh, but it could also be gas data.

We can stop to plot this data to get a better understanding of the general behavior of this meter.
```python
ax = df_baseline_108618['observed'].plot(label='Observed Usage', color='blue')
df_baseline_108618['temperature'].plot(ax=ax, secondary_y=True, label='Temperature (F)', color='orange')

ax.set_ylabel('Observed Usage (kWh)')
ax.right_ax.set_ylabel('Temperature (F)')

ax.legend(loc='upper left')
ax.right_ax.legend(loc='upper right')

plt.title('Observed Usage and Temperature in the Baseline Period')
plt.show()
```

??? Returns
    <div style="text-align: center; margin-top: 30px">
        <img src="../../images/eemeter/daily_billing/baseline_data_daily.png" alt="Daily Baseline Data">
    </div>

If we observe the data we can see a full year of data with observed usage peaking in the winter and lowering in the summer with warmer temperatures. It's clear that this site is located in a colder climate and uses more electricity in the winter.

### Loading Data into EEmeter Data Objects

With our sample data loaded into dataframes, we can create our Baseline and Reporting Data objects. Note that only the baseline period is needed to fit a model, but we will use our reporting period data to predict against.

```python
baseline_data = eemeter.eemeter.DailyBaselineData(df_baseline_108618, is_electricity_data=True)
reporting_data = eemeter.eemeter.DailyReportingData(df_reporting_108618, is_electricity_data=True)
```

These classes are critical to ensure standardized data loaded into the model, and they also scan the data to check for data sufficiency and other criteria that might cause a model to be disqualified (unable to build a model of sufficient integrity).


As a note, you can also instantiate these data classes with two separate meter usage and temperature Series objects, both indexed by timezone-aware datetime.

??? Example
    ```python
    baseline_data = eemeter.eemeter.DailyBaselineData.from_series(df_baseline_108618['observed'], df_baseline_108618['temperature'], is_electricity_data=True)
    ```

With data classes successfully instantiated, we can also check for any disqualifications or warnings before moving on to the model fitting step.

```python
print(f"Disqualifications: {baseline_data.disqualification}")
print(f"Warnings:          {baseline_data.warnings}")
```

??? Returns
    ```
    Disqualifications: []
    Warnings:          [EEMeterWarning(qualified_name=eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency), 
                        EEMeterWarning(qualified_name=eemeter.sufficiency_criteria.extreme_values_detected)]
    ```

From this, we can see that no disqualifications are present but there are some warnings to be aware of as we proceed. Neither of these warnings will necessarily stop us from creating a model.

Before we move on, also notice that you can access the underlying dataframe in each object like follows to see exactly what will be loaded into the model.

```python
print(baseline_data.df.head())
```

??? Returns
    ```
    datetime				    season	weekday_weekend	temperature	observed
    2018-01-01 00:00:00-06:00	winter	weekday	        -2.384038	16635.193673
    2018-01-02 00:00:00-06:00	winter	weekday	        1.730000	15594.051162
    2018-01-03 00:00:00-06:00	winter	weekday	        13.087946	11928.025899
    2018-01-04 00:00:00-06:00	winter	weekday	        4.743269	14399.333812
    2018-01-05 00:00:00-06:00	winter	weekday	        4.130577	14315.101721
    ```

### Creating the Model

The daily model follows the general process of:
1. Initialize
2. Fit
3. Predict

We can do this easily as follows:

```python
daily_model = eemeter.eemeter.DailyModel()
daily_model.fit(baseline_data)
```

Before we move to predicting against a dataframe, we can actually use the built in plot function (requiring matplotlib) to plot the performance of the model against the provided data.

```python
daily_model.plot(baseline_data)
```

??? Returns
    <div style="text-align: center; margin-top: 30px">
        <img src="../../images/eemeter/daily_billing/daily_baseline_vs_model.png" alt="Daily Baseline Observed vs. Model">
    </div>

From this graph we can also observe model splits and model types as described in the [Model Splits](../documentation/eemeter/billing_model/methodology/#model-splits) section. We can observe the following models:

1. Shoulder/Winter - Weekday
2. Summer - Weekday
3. Summer/Shoulder/Winter - Weekend

This illustrates that Summer/Shoulder/Winter weekends were similar enough to be modeled together, but Summer weekdays and Shoulder/Winter weekdays were different enough to require separate models. All of this complexity is handled under the hood, and the model will utilize the correct model when predicting usage automatically.

We can also use this function to plot the model prediction against the reporting period as follows:

```python
daily_model.plot(reporting_data)
```

??? Returns
    <div style="text-align: center; margin-top: 30px">
        <img src="../../images/eemeter/daily_billing/daily_reporting_vs_model.png" alt="Daily Reporting Observed vs. Model">
    </div>

In this plot we can see that the site is using significantly less energy in colder temperatures compared to the model / baseline period. Perhaps this site installed an efficiency intervention that saves energy in colder temperatures?

### Predicting with the Model and Calculating Savings

With our fit model, we can now predict across a given reporting period as follows:

```python
df_results = daily_model.predict(reporting_data)
print(df_results.head())
```

??? Returns
    ```
    datetime                    season	day_of_week	weekday_weekend	temperature	observed	predicted	    predicted_unc	heating_load	cooling_load	model_split	model_type											
    2019-01-01 00:00:00-06:00	winter	2	        weekday	        -2.384038	9294.220619	15610.330791	1181.674285	    14344.873494	0.0	            wd-sh_wi	hdd_tidd_cdd_smooth
    2019-01-02 00:00:00-06:00	winter	3	        weekday	        1.730000	8073.766329	14464.613486	1181.674285	    13199.156189	0.0	            wd-sh_wi	hdd_tidd_cdd_smooth
    2019-01-03 00:00:00-06:00	winter	4	        weekday	        13.087946	5261.174665	11322.966704	1181.674285	    10057.509407	0.0	            wd-sh_wi	hdd_tidd_cdd_smooth
    2019-01-04 00:00:00-06:00	winter	5	        weekday	        4.743269	6775.499525	13627.487003	1181.674285	    12362.029706	0.0	            wd-sh_wi	hdd_tidd_cdd_smooth
    2019-01-05 00:00:00-06:00	winter	6	        weekend	        4.130577	6735.513000	11690.139993	1224.574703	    11200.780385	0.0	            we-su_sh_wi	hdd_tidd_cdd_smooth
    ```

We can also plot the observed usage vs. the predicted usage.

```python
ax = df_results['observed'].plot(label='Observed Usage', color='blue')
df_results['predicted'].plot(ax=ax, label='Predicted Usage', color='orange')

ax.set_ylabel('Observed Usage (kWh)')
ax.legend(loc='upper left')
plt.title('Observed Usage and Temperature in the Baseline Period')
plt.savefig('predicted_vs_observed_daily.png')
plt.show()
```

??? Returns
    <div style="text-align: center; margin-top: 30px">
        <img src="../../images/eemeter/daily_billing/predicted_vs_observed_daily.png" alt="Daily Reporting Observed vs. Model">
    </div>

From here, we can easily calculate savings by subtracting observed usage from predicted usage.

```python
df_results['savings'] = df_results['predicted'] - df_results['observed']
print(f"Predicted Usage (kWh):  {round(df_results['predicted'].sum(), 2)}")
print(f"Observed Usage (kWh):   {round(df_results['observed'].sum(), 2)}")
print(f"Savings (kWh):          {round(df_results['savings'].sum(), 2)}")
```

??? Returns
    ```
    Predicted Usage (kWh):  1297651.07
    Observed Usage (kWh):   632077.62
    Savings (kWh):          665573.45
    ```

### Model Serialization

After creating a model, we can also serialize it for storage and read it back in later.

```python
saved_model = daily_model.to_json()
print(saved_model)
```

??? Returns
    ```python
    {
    "submodels": {
        "wd-su": {
        "coefficients": {
            "model_type": "tidd_cdd_smooth",
            "intercept": 1036.62605016263,
            "hdd_bp": null,
            "hdd_beta": null,
            "hdd_k": null,
            "cdd_bp": 63.112946428571426,
            "cdd_beta": 55.8938901507189,
            "cdd_k": 7.610345707770536
        },
        "temperature_constraints": {
            "T_min": 56.096346153846156,
            "T_max": 85.60365384615385,
            "T_min_seg": 63.112946428571426,
            "T_max_seg": 81.97444606863725
        },
        "f_unc": 287.2717928729389
        },
        "wd-sh_wi": {
        "coefficients": {
            "model_type": "hdd_tidd_cdd_smooth",
            "intercept": 1265.4572967952768,
            "hdd_bp": 48.66708982684446,
            "hdd_beta": 280.56118289086174,
            "hdd_k": 0.4861025625521579,
            "cdd_bp": 74.4246153846154,
            "cdd_beta": 76.12622638483967,
            "cdd_k": 0.2941773382321021
        },
        "temperature_constraints": {
            "T_min": -2.3840384615384616,
            "T_max": 79.53846153846153,
            "T_min_seg": 13.432307692307694,
            "T_max_seg": 74.4246153846154
        },
        "f_unc": 1181.6742853434923
        },
        "we-su_sh_wi": {
        "coefficients": {
            "model_type": "hdd_tidd_cdd_smooth",
            "intercept": 489.35960795932726,
            "hdd_bp": 51.43882581026119,
            "hdd_beta": 236.45604355078808,
            "hdd_k": 0.4856836346837153,
            "cdd_bp": 74.55925252366727,
            "cdd_beta": 66.31199103689823,
            "cdd_k": 0.36878985440304235
        },
        "temperature_constraints": {
            "T_min": 6.881923076923076,
            "T_max": 87.51442307692308,
            "T_min_seg": 23.602702266483515,
            "T_max_seg": 81.8328021978022
        },
        "f_unc": 1224.5747026357867
        }
    },
    "info": {
        "error": {
        "wRMSE": 580.7962073579329,
        "RMSE": 580.7962073579328,
        "MAE": 380.9965311661661,
        "CVRMSE": 0.16332929359071596,
        "PNRMSE": 0.06465498558194696
        },
        "baseline_timezone": "America/Chicago",
        "disqualification": [],
        "warnings": [
        {
            "qualified_name": "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
            "description": "Cannot confirm that pre-aggregated temperature data had sufficient hours kept",
            "data": {}
        },
        {
            "qualified_name": "eemeter.sufficiency_criteria.extreme_values_detected",
            "description": "Extreme values (greater than (median + (3 * IQR)), must be flagged for manual review.",
            "data": {
            "n_extreme_values": 2,
            "median": 2527.7263175451776,
            "upper_quantile": 5347.202740928209,
            "lower_quantile": 1247.7751658179252,
            "extreme_value_limit": 14826.00904287603,
            "max_value": 16635.193672683698
            }
        }
        ]
    },
    "settings": {
        "algorithm_choice": "nlopt_sbplx",
        "allow_separate_shoulder": true,
        "allow_separate_summer": true,
        "allow_separate_weekday_weekend": true,
        "allow_separate_winter": true,
        "alpha_final": "adaptive",
        "alpha_final_type": "last",
        "alpha_minimum": -100.0,
        "alpha_selection": 2.0,
        "cvrmse_threshold": 1.0,
        "developer_mode": false,
        "final_bounds_scalar": 1.0,
        "full_model": "hdd_tidd_cdd",
        "initial_guess_algorithm_choice": "nlopt_direct",
        "initial_smoothing_parameter": 0.5,
        "initial_step_percentage": 0.1,
        "is_weekday": {
        "1": true,
        "2": true,
        "3": true,
        "4": true,
        "5": true,
        "6": false,
        "7": false
        },
        "maximum_slope_OoM_scaler": 2.0,
        "reduce_splits_by_gaussian": true,
        "reduce_splits_num_std": [
        1.4,
        0.89
        ],
        "regularization_alpha": 0.001,
        "regularization_percent_lasso": 1.0,
        "season": {
        "1": "winter",
        "2": "winter",
        "3": "shoulder",
        "4": "shoulder",
        "5": "shoulder",
        "6": "summer",
        "7": "summer",
        "8": "summer",
        "9": "summer",
        "10": "shoulder",
        "11": "winter",
        "12": "winter"
        },
        "segment_minimum_count": 6,
        "smoothed_model": true,
        "split_selection_criteria": "bic",
        "split_selection_penalty_multiplier": 0.24,
        "split_selection_penalty_power": 2.061,
        "uncertainty_alpha": 0.1
    }
    }
    ```

Afterwards, we can instantiate the model as follows:

```python
loaded_model = eemeter.eemeter.DailyModel.from_json(saved_model)
```

## Billing Data / Model Differences

The Daily Model section generally applies to billing data as the same underlying model is used, but there are some key things to be aware of. It is strongly recommended to read the [daily model](#using-the-daily-model) section before this section.

### Loading Example Data

Like the daily model, we will start by loading some specific billing data.

```python
df_baseline, df_reporting = eemeter.load_test_data("monthly_treatment_data")
```

If we inspect these dataframes, we will notice that there are 100 meters for you to experiment with, indexed by meter id and datetime. Like before, we will filter to a single meter.

```python
df_baseline_108618 = df_baseline.loc[108618]
df_reporting_108618 = df_reporting.loc[108618]
```

If we inspect one of these dataframes, we will now notice that only a single meter is present with 365 days of data in each dataframe. 

```python
print(df_baseline_108618)
```

??? Returns
    ```
    id      datetime                     temperature     observed                       
    108618  2018-01-01 00:00:00-06:00    -2.384038  257406.539278
            2018-01-02 00:00:00-06:00     1.730000            NaN
            2018-01-03 00:00:00-06:00    13.087946            NaN
            2018-01-04 00:00:00-06:00     4.743269            NaN
            2018-01-05 00:00:00-06:00     4.130577            NaN
    ...                                       ...            ...
    120841  2018-12-27 00:00:00-06:00    52.010625            NaN
            2018-12-28 00:00:00-06:00    35.270000            NaN
            2018-12-29 00:00:00-06:00    29.630000            NaN
            2018-12-30 00:00:00-06:00    34.250000            NaN
            2018-12-31 00:00:00-06:00    43.311250            NaN
    ```

However, notice that we only have one observed value per month, even though we still have daily temperature data. The observed usage in a given month will be spread evenly across all days in the month when we instantiate the data classes.

### Loading Data into EEmeter Data Objects

Like the daily model, we will instantiate data objects, but this time we will use the Billing variant.

```python
billing_baseline_data = eemeter.eemeter.BillingBaselineData(df_baseline_108618, is_electricity_data=True)
billing_reporting_data = eemeter.eemeter.BillingReportingData(df_reporting_108618, is_electricity_data=True)
```

Notice that the observed usage for each day is automatically spread evenly from the monthly usage provided.

??? Returns
    ```
    datetime				    season	weekday_weekend	temperature	observed
    2018-01-01 00:00:00-06:00	winter	weekday     	-2.384038	8303.436751
    2018-01-02 00:00:00-06:00	winter	weekday     	1.730000	8303.436751
    2018-01-03 00:00:00-06:00	winter	weekday     	13.087946	8303.436751
    2018-01-04 00:00:00-06:00	winter	weekday     	4.743269	8303.436751
    2018-01-05 00:00:00-06:00	winter	weekday     	4.130577	8303.436751
    ...	...	...	...	...
    2018-12-27 00:00:00-06:00	winter	weekday     	46.602066	5288.700172
    2018-12-28 00:00:00-06:00	winter	weekday     	38.346724	5288.700172
    2018-12-29 00:00:00-06:00	winter	weekend     	28.614456	5288.700172
    2018-12-30 00:00:00-06:00	winter	weekend     	29.186923	5288.700172
    2018-12-31 00:00:00-06:00	winter	weekday     	36.510441	5288.700172
    ```

### Creating the Model and Predicting

With the data classes instantiated, we can now fit and predict like normal.

```python
billing_model = eemeter.eemeter.BillingModel().fit(billing_baseline_data, ignore_disqualification=False)
print(billing_model.predict(billing_reporting_data).head())
```

??? Returns
    ```
    datetime					season	day_of_week	weekday_weekend	temperature	observed	predicted	predicted_unc	heating_load	cooling_load	model_split	model_type						
    2019-01-01 00:00:00-06:00	winter	2	        weekday	        -2.384038	3655.09121	9733.728179	2316.237742 	8530.919994	    0.0	            fw-su_sh_wi	hdd_tidd_cdd
    2019-01-02 00:00:00-06:00	winter	3	        weekday	        1.730000	3655.09121	9235.507988	2316.237742	    8032.699803	    0.0	            fw-su_sh_wi	hdd_tidd_cdd
    2019-01-03 00:00:00-06:00	winter	4	        weekday	        13.087946	3655.09121	7860.032700	2316.237742 	6657.224516	    0.0	            fw-su_sh_wi	hdd_tidd_cdd
    2019-01-04 00:00:00-06:00	winter	5	        weekday	        4.743269	3655.09121	8870.593662	2316.237742	    7667.785478	    0.0	            fw-su_sh_wi	hdd_tidd_cdd
    2019-01-05 00:00:00-06:00	winter	6	        weekend	        4.130577	3655.09121	8944.792210	2316.237742	    7741.984025	    0.0	            fw-su_sh_wi	hdd_tidd_cdd
    ```

Notice that the returned data is at the daily level, showing daily predictions and observed values. We can also aggregate this data up to a higher level as follows.

```python
print(billing_model.predict(billing_reporting_data, aggregation="monthly"))
```

??? Returns
    ```
    datetime	                season	    temperature	observed	    predicted	    predicted_unc	heating_load	cooling_load	model_split	model_type								
    2019-01-01 00:00:00-06:00	winter	    25.612211	113307.827517	196642.717137	12896.265955	159355.663417	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    2019-02-01 00:00:00-06:00	winter	    29.523429	81056.766022	164350.346292	12256.378085	130671.717126	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    2019-03-01 00:00:00-06:00	shoulder	36.122525	62783.320515	157185.144004	12896.265955	119898.090284	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    2019-04-01 00:00:00-05:00	shoulder	40.599556	53542.223426	135849.270725	12686.556598	99765.025189	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    2019-05-01 00:00:00-05:00	shoulder	64.969863	31089.429512	59427.613500	12896.265955	22047.507990	93.051790	    fw-su_sh_wi	hdd_tidd_cdd
    2019-06-01 00:00:00-05:00	summer	    70.262968	29346.533659	43017.949048	12686.556598	6801.829282	    131.874231	    fw-su_sh_wi	hdd_tidd_cdd
    2019-07-01 00:00:00-05:00	summer	    75.915400	32743.001296	37549.881062	12896.265955	0.000000	    262.827342	    fw-su_sh_wi	hdd_tidd_cdd
    2019-08-01 00:00:00-05:00	summer	    75.706736	33660.542162	38028.046427	12896.265955	480.770213	    260.222494	    fw-su_sh_wi	hdd_tidd_cdd
    2019-09-01 00:00:00-05:00	summer	    68.480234	30381.828023	46721.644687	12686.556598	10530.068712	107.330440	    fw-su_sh_wi	hdd_tidd_cdd
    2019-10-01 00:00:00-05:00	shoulder	52.733661	34995.917728	97491.932913	12896.265955	60181.378730	23.500463	    fw-su_sh_wi	hdd_tidd_cdd
    2019-11-01 00:00:00-05:00	winter	    35.239608	63200.887271	155322.359404	12686.556598	119238.113869	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    2019-12-01 00:00:00-06:00	winter	    33.696016	65969.341170	166294.687445	12896.265955	129007.633726	0.000000	    fw-su_sh_wi	hdd_tidd_cdd
    ```

With the monthly aggregation, the resulting dataframe is aggregated by month and 12 rows are returned.