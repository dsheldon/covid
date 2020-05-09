import configs
import covid.util as util
import covid.models.SEIRD_variable_detection
import numpy as np
import pandas as pd

from pathlib import Path

data = util.load_state_data()

#config_names = ['SEIRD', 'SEIRD_incident', 'SEIRD_variable_detection']
#forecast_dates = ["2020-04-04", "2020-04-11",  "2020-04-18", "2020-04-25"]

config_names = ['strongest_prior']
forecast_dates = ['2020-04-25', '2020-05-03']
eval_date = '2020-05-06'
root='results'

overall_summary = pd.DataFrame()

for config_name in config_names:
    print(f"****Config {config_name}****")
    
    for forecast_date in forecast_dates:
        print(f" **Forecast date {forecast_date}**")

        forecast_start = pd.to_datetime(forecast_date) + pd.Timedelta("1d")
        prefix = f"{root}/{config_name}/{forecast_date}"

        config = getattr(configs, config_name)          
        errs = []

        summary, details = util.score_forecast(forecast_start,
                                               data,
                                               model_type=config['model'],
                                               prefix=prefix)
        

        summary.insert(0, 'model', config_name)
        details.insert(0, 'model', config_name)
        
        path = Path(prefix) / 'eval'
        path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path / 'summary.csv', float_format="%.4f")
        details.to_csv(path / 'details.csv', float_format="%.4f")
        
        overall_summary = overall_summary.append(summary.loc[eval_date])
        

filename = Path(root) / 'summary.csv'
overall_summary.to_csv(filename, float_format="%.4f", index=False)
print(overall_summary)