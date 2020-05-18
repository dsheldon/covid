import configs
import covid.util as util
import covid.models.SEIRD_variable_detection
import numpy as np
import pandas as pd

from pathlib import Path

data = util.load_state_data()

config_names=['fit_dispersion']
forecast_dates = ['2020-04-18', '2020-04-25', '2020-05-03', '2020-05-10']
eval_date = '2020-05-16'
root='results1'

def write_summary(summary, filename):
    summary = summary.reset_index(drop=True)
    cols = list(summary.columns)
    special_cols = ['model', 'forecast_date', 'eval_date', 'horizon']
    for c in special_cols:
        cols.remove(c)    
    cols = special_cols + cols
    summary.to_csv(filename, float_format="%.4f", columns=cols, index=False)


overall_summary = pd.DataFrame()

for config_name in config_names:
    print(f"****Config {config_name}****")

    config_summary = pd.DataFrame()
    
    for forecast_date in forecast_dates:
        print(f" **Forecast date {forecast_date}**")

        prefix = f"{root}/{config_name}/{forecast_date}"

        config = getattr(configs, config_name)          
        errs = []

        summary, details = util.score_forecast(forecast_date,
                                               data,
                                               model_type=config['model'],
                                               prefix=prefix)
        

        summary.insert(0, 'model', config_name)
        details.insert(0, 'model', config_name)
        
        path = Path(prefix) / 'eval'
        path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path / 'summary.csv', float_format="%.4f")
        details.to_csv(path / 'details.csv', float_format="%.4f")


        config_summary = config_summary.append(summary.loc[eval_date])
        

    # add eval date and save
    config_summary['eval_date'] = eval_date

    print(f"***Config {config_name} results***")
    print(config_summary)
    write_summary(config_summary, f"{root}/{config_name}/summary.csv")

    overall_summary = overall_summary.append(config_summary)
    

# write overall summary
write_summary(overall_summary, Path(root) / 'summary.csv')
print(f"***Overall results***")
print(overall_summary)

