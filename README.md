# acbm

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A package to create activity-based models (for transport demand modelling)

- [acbm](#acbm)
- [Motivation and Contribution](#motivation-and-contribution)
- [Installation](#installation)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
  - [Step 1: Prepare Data Inputs](#step-1-prepare-data-inputs)
  - [Step 2: Setup your config.toml file](#step-2-setup-your-configtoml-file)
  - [Step 3: Run the pipeline](#step-3-run-the-pipeline)
  - [Future Work](#future-work)
    - [Generative Aproaches to activity scheduling](#generative-aproaches-to-activity-scheduling)
    - [Location Choice](#location-choice)
  - [Related Work](#related-work)
    - [Synthetic Population Generation](#synthetic-population-generation)
    - [Activity Generation](#activity-generation)
      - [Deep Learning](#deep-learning)
    - [Location Choice](#location-choice-1)
      - [Primary Locations](#primary-locations)
      - [Secondary Locations](#secondary-locations)
    - [Entire Pipeline](#entire-pipeline)
- [Contributing](#contributing)
- [License](#license)


# Motivation and Contribution

Activity-based models have emerged as an alternative to traditional 4-step transport demand models. They provide a more detailed framework by modeling travel as a sequence of activities, accounting for when, how, and with whom individuals participate. They can integrate household interactions, spatial-temporal constraints, are well suited to model on demand transport services (which are becoming increasingly common), and look at the equity implications across transport scenarios.

Despite being increasingly popular in research, adoption in industry has been slow. A couple of factors have influenced this. The first is inertia and well-established guidelines on 4-step models. However, this is changing; in 2024, the UK Department for Transport released its first Transport Analysis Guidance on activity and agent-based models (See [TAG unit M5-4 agent-based methods and activity-based demand modelling](<ins>https://www.gov.uk/government/publications/tag-unit-m5-4-agent-based-methods-and-activity-based-demand-modelling</ins>)). Other initiatives, such as the [European Association of Activity-Based Modeling](https://eaabm.org/) are also being established to try and increase adoption of activity-based modelling and push research into practice.

Another factor is tool availability. Activity-based modelling involves many steps, including synthetic population generation, activity sequence generation, and (primary and secondary) location assignment. Many tools exist for serving different steps, but only a couple of tools exist to run an entire, configurable pipeline, and they tend to be suited to the data of specific countries (see [Related Work](<ins>#related-work</ins>) for a list of different open-source tools).

To our knowledge, no open-source activity-based modelling pipeline exists for the UK. This repository allows researchers to run the entire pipeline for any region in the UK, with the output being a synthetic population with daily activity diaries and locations for each person. The pipeline is meant to be extendible, and we aim to plug in different approaches developed by others in the future

# Installation

```bash
python -m pip install acbm
```

From source:
```bash
git clone https://github.com/alan-turing-institute/acbm
cd acbm
poetry install
```

# How to Run the Pipeline

The pipeline is a series of scripts that are run in sequence to generate the activity-based model. There are a few external datasets that are required. The data and config directories are structured as follows:

```md
├── config
│   ├── <your_config_1>.toml
│   ├── <your_config_2>.toml
├── data
│   ├── external
│   │   ├── boundaries
│   │   │   ├── MSOA_DEC_2021_EW_NC_v3.geojson
│   │   │   ├── oa_england.geojson
│   │   │   ├── study_area_zones.geojson
│   │   ├── census_2011_rural_urban.csv
│   │   ├── centroids
│   │   │   ├── LSOA_Dec_2011_PWC_in_England_and_Wales_2022.csv
│   │   │   └── Output_Areas_Dec_2011_PWC_2022.csv
│   │   ├── MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv
│   │   ├── nts
│   │   │   ├── filtered
│   │   │   │   ├── nts_households.parquet
│   │   │   │   ├── nts_individuals.parquet
│   │   │   │   └── nts_trips.parquet
│   │   │   └── UKDA-5340-tab
│   │   │       ├── 5340_file_information.rtf
│   │   │       ├── mrdoc
│   │   │       │   ├── excel
│   │   │       │   ├── pdf
│   │   │       │   ├── UKDA
│   │   │       │   └── ukda_data_dictionaries.zip
│   │   │       └── tab
│   │   │           ├── household_eul_2002-2022.tab
│   │   │           ├── individual_eul_2002-2022.tab
│   │   │           ├── psu_eul_2002-2022.tab
│   │   │           ├── trip_eul_2002-2022.tab
│   │   │           └── <other_nts_tables>.tab
│   │   ├── travel_times
│   │   │    ├── oa
│   │   │    |   ├── travel_time_matrix.parquet
|   |   │    └── msoa
│   │   │        └── travel_time_matrix.parquet
│   │   ├── ODWP01EW_OA.zip
│   │   ├── ODWP15EW_MSOA_v1.zip
│   │   └── spc_output
│   │      └── raw
│   │          ├── <region>_households.parquet
│   │          ├── <region>_info_per_msoa.json
│   │          ├── <region>.pb
│   │          ├── <region>_people.parquet
│   │          ├── <region>_time_use_diaries.parquet
│   │          ├── <region>_venues.parquet
│   │          └── README.md
│   └── outputs
│       └- <config_id>
│           │  
│           ├── interim
│           │       ├── <region>>_people_hh.parquet (Generated in Script 1)
│           │       ├── assigning (Generated in Script 3)
│           │       └── matching (Generated in Script 2)
│           ├── activities.csv
│           ├── households.csv
│           ├── legs.csv
│           ├── legs_with_locations.parquet
│           ├── people.csv
│           ├── plans.xml
│           ├── plots
│           │   ├── assigning
│           │   └── validation
```

## Step 1: Prepare Data Inputs

You need to populate the data/external diectory with the required datasets. A guide on where to find / generate each dataset can be found in the [data/external/README.md](https://github.com/Urban-Analytics-Technology-Platform/acbm/tree/main/data/external)

## Step 2: Setup your config.toml file

You need to create a config file in the config directory. The config file is a toml file that contains the parameters for the pipeline. A guide on how to set up the config file can be found in the [config/README.md](https://github.com/Urban-Analytics-Technology-Platform/acbm/tree/main/config)

## Step 3: Run the pipeline

The scripts are listed in order of execution in the [scripts/run_pipeline.sh](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/run_pipeline.sh) bash file

You can run the pipeline by executing the following command in the terminal from the base directory:

```bash
bash ./scripts/run_pipeline.sh config/<your_config_file>.toml
```

where your config file is the file you created in Step 2.

For details on what each script does, you can check the [scrips/README.md](https://github.com/Urban-Analytics-Technology-Platform/acbm/tree/main/scripts) and the [WIKI](https://github.com/Urban-Analytics-Technology-Platform/acbm/wiki)

## Future Work

We aim to include different options for each step of the pipeline. Some hopes for the future include:

### Generative Aproaches to activity scheduling
- [ ] Bayesian Network approach to generate activities
- [ ] Implement a Deep Learning approach to generate activities (see package below)

### Location Choice
- [ ] Workzone assignment: Plug in Neural Spatial Interaction Approach

## Related Work

There are a number of open-source tools for different parts of the activity-based modelling pipeline. Some of these include:

### Synthetic Population Generation

### Activity Generation

#### Deep Learning
- [caveat](https://github.com/fredshone/caveat)

### Location Choice

#### Primary Locations

- [GeNSIT](https://github.com/YannisZa/GeNSIT)

#### Secondary Locations
- [PAM](https://github.com/arup-group/pam/blob/main/examples/17_advanced_discretionary_locations.ipynb)


### Entire Pipeline
- [Eqasim](https://github.com/eqasim-org/eqasim-java)
- [ActivitySim](https://activitysim.github.io/activitysim/v1.3.1/index.html)
- [PAM](https://github.com/arup-group/pam): PAM has functionality for different parts of the pipeline, but itis not clear how to use it to create an activity-based model for an entire population. Specifically, it does not yet have functionality for activity generation (e.g. statistical matching or generative approaches), or constrained primary location assignment.


# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

# License

Distributed under the terms of the [Apache license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/acbm/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/acbm/actions
[pypi-link]:                https://pypi.org/project/acbm/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/acbm
[pypi-version]:             https://img.shields.io/pypi/v/acbm
<!-- prettier-ignore-end -->
