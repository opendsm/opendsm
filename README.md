# OpenDSM: Tools for calculating metered energy savings

[![PyPI Version](https://img.shields.io/pypi/v/opendsm.svg)](https://pypi.python.org/pypi/opendsm)
[![Supported Versions](https://img.shields.io/pypi/pyversions/opendsm.svg)](https://github.com/opendsm/opendsm)
[![License](https://img.shields.io/github/license/opendsm/opendsm.svg)](https://github.com/opendsm/opendsm)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

---------------

**OpenDSM (formerly OpenEEmeter)** — an open-source library used to measure the impacts 
of demand-side programs by using historical data to fit models and then create 
predictions (counterfactuals) to compare to post-intervention, observed energy usage.

## Background - Why use OpenDSM

Energy efficiency programs have traditionally focused on addressing long-term load growth 
and reducing customer energy bills rather than serving as reliable grid resources. 
However, as utilities work to decarbonize power generation, buildings, and transportation, 
demand-side programs (e.g. energy efficiency, load shifting, electrification, and demand 
response programs) must evolve into dependable, scalable grid assets. Ultimately, 
decarbonizing the power grid will require both supply and demand-side solutions. While 
supply-side production is easily quantified, measuring the impacts of demand-side programs 
has historically been challenging due to inconsistent and opaque measurement methodologies.

OpenDSM solves these problems with accurate, efficient, and transparent models designed to 
measure demand-side program impacts. OpenDSM gives all stakeholders full visibility and 
confidence in the results.

OpenDSM builds upon the shoulders of OpenEEmeter and the [CalTRACK Methods](https://caltrack.org/) which themselves
are built upon the foundational work of the Princeton Scorekeeping Method ([PRISM 1986](https://www.marean.mycpanel.princeton.edu/images/prism_intro.pdf)) 
for the daily and billing models and Lawrence Berkeley National Laboratory's Time-of-Week 
and Temperature Model ([TOWT 2011](https://eta-publications.lbl.gov/sites/default/files/LBNL-4944E.pdf)) for the hourly energy efficiency and demand response models.
OpenDSM models have been proven to meet or exceed the predictive capablity of the 
aforementioned models. These models adhere to a statistical approach, as opposed to an 
engineering approach, so that these models can be efficiently run on millions of meters at 
a time, while still providing accurate predictions. 

Using default settings in OpenDSM will provide accurate and stable model predictions 
suitable for savings measurements from demand side interventions. Settings can be modified 
and sufficiency requirements can be bypassed for research and development purposes; however, 
the outputs of such models are no longer OpenDSM compliant measurements as the modifications
mean that these models are no longer verified and approved by the OpenDSM Working Group.

## Installation

OpenDSM is a python package and can be installed with pip.

~~~~~~~~~~~~~~~
$ pip install opendsm
~~~~~~~~~~~~~~~

## Features

- Models:

  - Energy Efficiency Daily Model
  - Energy Efficiency Billing (Monthly) Model
  - Energy Efficiency Non-Solar Hourly Model
  - Energy Efficiency Solar Hourly Model
  - Demand Response Hourly Model

- Flexible sources of temperature data. See [EEweather](https://github.com/opendsm/eeweather).
- Data sufficiency checking
- Model serialization
- First-class warnings reporting

## [Documentation](https://opendsm.energy/)

Documenation for this library can be found [here](https://opendsm.energy/).

## Future Development

The OpenDSM project growth goals fall into two categories:

1. Community goals - we want help our community thrive and continue to grow.
2. Technical goals - we want to keep building the library in new ways that make it
   as easy as possible to use.

### Community goals

1. Improve repository structure, architecture, and API

The first step of being able to contribute to a project is to understand how the repository
is laid out and how OpenDSM is architected. We have made giant steps in this area as of late, 
but there is additional organizational work to be done. This will continue to be an ongoing
area of work.

2. Make it easier to contribute

As our user base grows, the need and desire for users to contribute back to the library
also grows, and we want to make this as seamless as possible. This means writing and
maintaining contribution guides, and creating checklists to guide users through the
process.


### Technical goals

1. Update the Demand Response (DR) model

In the most recent release, the hourly energy efficiency (EE) model has been entirely
changed and updated. Much like the billing model is to the daily model, the DR model is a
subset of the EE hourly model. Many of the improvements seen in the EE hourly model could
be realized in the DR model if it were finalized. It is currently in a functional state
within a branch, but its parameters have not been optimized rendering it unusable for
measurements. In the meantime, the existing DR model is still available.

2. Reassess existing sufficiency and disqualification criteria

The existing sufficiency and disqualification criteria exist as conservative estimates
from OpenEEmeter and CalTRACK recommendations. There is almost certainly room for these
criteria to be revisited so that more meters would pass and be approved for measurement.

3. Determine the sufficiency requirements of PV installation date in the hourly model

The hourly EE model currently has the capability of ingesting a PV installation date and
generating an additional feature that can much better represent a meter who installs a
solar PV system mid-baseline year. However, this feature currently is classified as
experimental and not allowed for official measurement because we have not quantified how
much data is required post-installation to be able to accurately predict the meter's
behavior in the reporting year.

4. Improve the daily model

There are two potential areas of improvement of the daily model. First it could be extended
to allow additional sources of information, but this must carefully be considered as the
primary usage of the daily model is to be able to disaggregate heating and cooling usage.
The second area of improvement would be to allow an additional break point within both the
cooling and heating regions such that the model would be able to change slope. This should
likely still be limited such that the model's slope in each region is appropriately
constrained. A new smoothing function would also need to be developed.

5. Integrate EEweather

EEweather is commonly used to obtain weather information to be used within OpenDSM. If it 
were more tightly coupled, it would streamline the most standard use of OpenDSM. As an 
example this could simplify several of the data classes such that the aggregation of 
weather data would be done within EEweather instead of within data classes where it is a
more complex procedure

6. Integrate GRIDmeter

GRIDmeter is frequently used after DR/EEmeter in order to correct models for external
population-level effects by using non-participant meters. Similarly to EEweather, this 
process could be streamlines and made more cohesive by fully integrating it into OpenDSM. 

7. Organize and revise existing test suite

The existing testing suite is the last remaining vestige of the library prior to the 
extensive reorganization and API changes made. It would be well served to update the
testing suite to make it easier for future contributors to know how and where they 
should develop their tests for any new features or bugs found.

8. Greater weather coverage

The weather station coverage in the EEweather package includes full coverage of US and
Australia, but with some technical work, it could be expanded to include greater, or
even worldwide coverage.

## License

This project is licensed under [Apache 2.0](LICENSE).

## Other resources

- [CONTRIBUTING](https://github.com/opendsm/opendsm/blob/master/CONTRIBUTING.md): How to contribute to the project.
- [MAINTAINERS](https://github.com/opendsm/opendsm/blob/master/MAINTAINERS.md): An ordered list of project maintainers.
- [CHARTER](https://github.com/opendsm/opendsm/blob/master/CHARTER.md): Open source project charter.
- [CODE OF CONDUCT](https://github.com/opendsm/opendsm/blob/master/CODE_OF_CONDUCT.md): Code of conduct for contributors.


