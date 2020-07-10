import configs
import covid.util as util
import covid.states as states
import covid.models.SEIRD_variable_detection
import covid.jhu as jhu
import numpy as np
import pandas as pd
import argparse

from pathlib import Path

county_list = list(jhu.get_county_info().sort_values('Population', ascending=False).index)
state_list = list(jhu.get_county_info().sort_index().index)

config_names=['counties']
forecast_dates = ['2020-06-21']
eval_date = '2020-07-03'
root='results1'
places=county_list[:500]

def write_summary(summary, filename):
    summary = summary.reset_index(drop=True)
    cols = list(summary.columns)
    special_cols = ['model', 'forecast_date', 'eval_date', 'horizon']
    for c in special_cols:
        cols.remove(c)    
    cols = special_cols + cols
    summary.to_csv(filename, float_format="%.4f", columns=cols, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score compartmental models.')
    parser.add_argument('places', help='places to use (e.g., US state)', nargs='*', )
    args = parser.parse_args()

    data = util.load_data()
    places = args.places or places

    # Special case for when places is only US
    suffix=''
    if places == ['US']:
        suffix = '-US'

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
                                               prefix=prefix,
                                               places=places   )
        

            summary.insert(0, 'model', config_name)
            details.insert(0, 'model', config_name)
        
            path = Path(prefix) / 'eval'
            path.mkdir(parents=True, exist_ok=True)
            summary.to_csv(path / f'summary{suffix}.csv', float_format="%.4f")
            details.to_csv(path / f'details{suffix}.csv', float_format="%.4f")


            config_summary = config_summary.append(summary.loc[eval_date])
        

        # add eval date and save
        config_summary['eval_date'] = eval_date

        print(f"***Config {config_name} results***")
        print(config_summary)
        write_summary(config_summary, f"{root}/{config_name}/summary{suffix}.csv")

        overall_summary = overall_summary.append(config_summary)
    

    # write overall summary
    write_summary(overall_summary, Path(root) / f'summary{suffix}.csv')
    print(f"***Overall results***")
    print(overall_summary)
