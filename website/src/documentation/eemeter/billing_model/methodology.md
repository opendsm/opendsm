The billing model uses the daily model with some slight configuration changes and uses billing (monthly or bimonthly) interval data.

## How the Model Works

### Converting from billing data to daily data

The billing model intakes billing data and calculates the average energy usage for each given period. For example if a bill covers 31 days and uses 93 therms total, then each day would be given 3 therms of usage. Pulling the temperature from `EEweather` ensures that each day's mean temperature is correct.

From here, the data is treated as daily interval data and uses the daily model internally.

### Model Shape and Balance Points

The daily model, at its core, utilizes a piecewise linear regression model that predicts energy usage relative to temperature. The model determines temperature balance points at which energy usage starts changing relative to temperature.

<div style="text-align: center">
    <img src="/../../../images/eemeter/billing_model/basic_model.png" alt="Billing Model">
</div>

#### Key Concepts

- **Balance Points**: Outdoor temperature thresholds beyond which heating and cooling effects are observed.
- **Heating and Cooling Coefficients**: Rate of increase of energy use per change in temperature beyond the balance points.
- **Temperature Independent Load**: The regression intercept (height of the flat line in the diagram).
- **Billing Data**: Because the billing data is converted to daily data, plots will show lines of data points where the usage is the same over a range of temperatures seen during that period.

#### Model Archetypes

Based on the site behavior, there are four different model types that may be generated:
- Heating and Cooling Loads
- Heating Only Load
- Cooling Only Load
- Temperature Independent Load

<div style="text-align: center; margin-top: 30px">
    <img src="/../../../images/eemeter/common/model_archetypes.png" alt="Different model archetypes">
</div>

#### Smooth Transitions

The billing model disables this feature.

#### Robust, Adaptive Outlier Downweighting

The billing model disables this feature.

#### Model Fit

When the model is fit, each site will receive its own unique model fit and coefficients. The general model fitting process is as follows:

1. Balance points are estimated with a global optimization algorithm.
2. Sum of Squares Error (SSE) is minimized with Lasso regression inspired penalization.
3. The best model type is determined (ex. cooling load only model) using the penalized SSE.

The Lasso inspired penalization means that increased model complexity must be justified by decreased SSE and balanced against these general rules:

- Slopes are pushed to 0
- Intercept is pushed to 0
- Balance points are pushed together
- Balance points are pushed towards the nearest edge (most extreme temperature)

<div style="text-align: center; margin-top: 30px">
    <img src="/../../../images/eemeter/billing_model/lasso_penalization.png" alt="Lasso penalization">
</div>

At this point the billing model is now fit and can be used for prediction.

### Model Splits

Unlike the standard daily model, the billing model is configured to not allow splits. There is not enough information to make these distinctions.

### Real Data Example

Here are 6 examples of how the billing model performs on real data.

<div style="text-align: center; margin-top: 30px">
    <img src="/../../../images/eemeter/billing_model/real_examples.png" alt="Real world examples">
</div>

*For additional information and validation details, see the [References](/../documentation/eemeter/billing_model/references/) page.