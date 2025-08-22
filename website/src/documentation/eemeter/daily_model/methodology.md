The daily model is trained using daily energy usage intervals and also predicts energy usage in daily intervals. The billing model is identical to the daily model, but allows users to train using billing interval data instead, while handling the daily usage distribution under the hood.

## How the Model Works

### Model Shape and Balance Points

The daily model, at its core, utilizes a piecewise linear regression model that predicts energy usage relative to temperature. The model determines temperature balance points at which energy usage starts changing relative to temperature.

<div style="text-align: center">
    <img src="../../images/eemeter/daily_billing/daily_model_balance_points.png" alt="Daily Model Balance Points">
</div>

The key terms to understand here are:

- **Balance Points**: Outdoor temperature thresholds beyond which heating and cooling effects are observed.
- **Heating and Cooling Coefficients**: Rate of increase of energy use per change in temperature beyond the balance points.
- **Temperature Independent Load**: The regression intercept (height of the flat line in the diagram).

Based on the site behavior, there are four different model types that may be generated:
- Heating and Cooling Loads
- Heating Only Load
- Cooling Only Load
- Temperature Independent Load

<div style="text-align: center; margin-top: 30px">
    <img src="../../images/eemeter/daily_billing/heating_cooling_load.png" alt="Different Model Types">
</div>

When the model is fit, each site will receive its own unique model fit and coefficients. The general model fitting process is as follows:

1. Balance points are estimated with a global optimization algorithm.
2. Sum of Squares Error (SSE) is minimized with Lasso regression inspired penalization.
3. The best model type is determined (ex. cooling load only model)
4. The model best fit is found using the penalized SSE.

### Model Splits

The process described above is effective but may have shortcomings in real life data if energy usage changes fundamentally during different time periods.

For example, what if a site is more populated during a particular season (for example, a Summer House or Ski Lodge) or during weekdays (for example, offices and most homes). This may result in models that fail to accurately predict energy usage because they are trying to account for all time periods at once.

<div style="display: flex; justify-content: center; margin-top: 30px">
    <img src="../../images/eemeter/daily_billing/season_problems.png" alt="Seasonal Misalignment" style="max-width: 50%">
    <img src="../../images/eemeter/daily_billing/weekday_problems.png" alt="Weekday Misalignment" style="max-width: 50%">
</div>

To combat this, the model will create "splits" that will store independent models for different seasons or weekday/weekend combinations, but only if necessary. 

The general process is as follows:

1. Create models using all possible splits of season/weekday|weekend.
2. Calculate modified BIC (Bayesian Information Criterion) for each preliminary combination.
3. Select combination with the smallest BICmod.
3. Best model type is inferred and best fit is found.

This provides a standardized process for splitting the model to better predict energy usage by certain time periods (if the benefit outweighs the additional model complexity).

<div style="text-align: center; margin-top: 30px">
    <img src="../../images/eemeter/daily_billing/split_model_season.png" alt="Model split by season">
</div>