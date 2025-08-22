::: opendsm.eemeter.models.daily
    options:
      show_submodules: true
      filters:
      - "!^_"
      - "!^from_2_0"
      - "!seasonal_options"
      - "!day_options"
      - "!combo_dictionary"
      - "!verbose"
      members:
      - DailyModel

::: opendsm.eemeter.models.daily.utilities.settings
    options:
      filters:
      - "!^__repr"
      - "!"
      members:
      - DailySettings
      
::: opendsm.eemeter.models.daily
    options:
      show_submodules: true
      filters:
      - "!^_"
      - "!log_warnings"
      members:
      - DailyBaselineData
      - DailyReportingData

example:
