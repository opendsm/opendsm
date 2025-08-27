---
hide:
 - toc
---

### Model Structure

Models are developed assuming that minimal information will be available for any given meter. The only assumptions made are that we will have a meter's location and usage data. Using the meter's location, its weather can be looked up using `EEweather`. This weather information can be used in place of a meter's location. 

Any additional information must be broadly available to be considered for addition into any OpenDSM models. This means that these models will inherently be statistical in nature, although informed by our engineering and domain knowledge.

Temperature inputs are expected to be in Fahrenheit. Failure to convert your units properly can result in erroneous models.

### Model Performance

Our models must be proven with extreme rigor to be superior to the prior model that they replace to be approved for measurement purposes. 

Improvements can come in the form of global accuracy, local accuracy, replicability, API improvements, and/or computational efficiency.