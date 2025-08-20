# OpenDSM

<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p align="center" id="opendsm">
  <a href="https://lfenergy.org/projects/opendsm/"><img src="./images/common/opendsm-horizontal-color.svg#only-light" alt="OpenDSM"></a>
  <a href="https://lfenergy.org/projects/opendsm/"><img src="./images/common/opendsm-horizontal-white.svg#only-dark" alt="OpenDSM"></a>
</p>

<p align="center">
    <em>OpenDSM: An open-source python package to develop and implement standard methods for predicting metered energy usage.</em>
</p>

</p>
    <p align="center">
    <a href="https://pypi.python.org/pypi/opendsm" target="_blank">
        <img src="https://img.shields.io/pypi/v/opendsm.svg" alt="PyPi Version">
    </a>
    <a href="https://pypi.org/project/opendsm" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/opendsm.svg" alt="Supported Python versions">
    </a>
    <a href="https://github.com/opendsm/opendsm" target="_blank">
        <img src="https://img.shields.io/github/license/opendsm/opendsm.svg" alt="License">
    </a>
    <a href="https://github.com/ambv/black" target="_blank">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style">
    </a>
</p>

---

**Source Code**: <a href="https://github.com/opendsm/opendsm" target="_blank">https://github.com/opendsm/opendsm</a>

---

OpenDSM (formerly OpenEEmeter) is an open source library used to measure the impacts of demand-side programs by using historical data to fit models and then create predictions (counterfactuals) to compare to post-intervention, observed energy usage.

## Key Features include:

- **Open Source**: All code is open source, making OpenDSM an excellent choice for measuring energy savings when results must be transparent and reproducible.
- **Fast**: A key tenet of OpenDSM modules are to be highly efficient, critical when there are millions of meters to be modeled. 
- **Easy and Intuitive**: The API interface is inspired by <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a>, a well-known data science package for building models. Just fit and predict.
- **Pandas DataFrame Support**: Input data and predictions use pandas DataFrames, a well-known format for data scientists and engineers.
- **Model Serialization and Deserialization**: Models can be serialized into dictionaries or json objects and deserialized later.

## Core Modules

### OpenEEmeter

Create models fit on historical data to generate predictions of energy consumption

- **Models to Fit Your Data**: Billing, daily, and hourly models, depending on the time-series resolution
- **Input Data Formatting**: Meter usage and temperature time series are input to models through data classes to ensure standardization and avoid common pitfalls
- **Data and Model Sufficiency**: Native sufficiency checking to verify measurement compliance with the approved methodology


### GRIDmeter

Assign comparison groups (CGs) to correct OpenEEmeter models using non-participant population through one of the following methods

- **Comparison Group Clustering**: Cluster on model error profiles to select unique CG for each treatment meter
- **Individual Meter Matching**: Create population-level corrections by choosing the nearest *n* meters using Euclidean distance
- **Stratified Sampling**: Select meters for CG based upon shared characteristics (outdated methdology)

### EEweather

Get the most appropriate weather data for a location

- **Match Location**: Select a weather station using latitude/longitude or ZIP code (ZCTA)
- **Climate zone sensitive**: Ensures that the selected weather station is within the same climate zone
- **Uses reliable sources**: Utilizes US Census Bureau, NOAA NCDC, and NREL as primary data sources