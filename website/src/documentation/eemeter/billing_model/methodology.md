The billing model uses the daily model with some slight configuration changes and uses billing (monthly or bimonthly) interval data.

## How the Model Works

### Converting from billing data to daily data

The billing model intakes billing data and calcualtes the average energy usage for each given period. For example if a bill covers 31 days and uses 93 therms total, then each day would be given 3 therms of usage. Pulling the temperature from `EEweather` ensures that each day's mean temperature is correct.

The data is now treated as daily interval data and uses the daily model internally.

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

At this point the billing model is now fit and can be used for prediction.

### Model Splits

Unlike the standard daily model, the billing model is configured to not allow splits. There is not enough real information to make these distinctions.