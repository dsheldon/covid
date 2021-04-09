import numpyro
numpyro.enable_x64()

import sys
import argparse
import covid.util as util
import configs
import numpy as onp
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Bayesian compartmental models.')
    parser.add_argument('place', help='place to use (e.g., US state)')
    parser.add_argument('--start', help='start date', default='2020-03-04')
    parser.add_argument('--end', help='end date', default=None)
    parser.add_argument('--prefix', help='path prefix for saving results', default='results')
    parser.add_argument('--no-run', help="don't run the model (only do vis)", dest='run', action='store_false')
    parser.add_argument('--config', help='model configuration name', default='SEIRD')
    parser.set_defaults(run=True)

    args = parser.parse_args()

    if args.config not in dir(configs):
        print(f'Invalid config: {args.config}. Options are {dir(configs)}')
        exit()

    config = getattr(configs, args.config)

<<<<<<< HEAD
    data = config.get('data') or util.load_data()

    # April 4: didn't report or had anomalous low counts on Easter Sunday
    for place in ['KY', 'MN', 'OH', 'SC', 'SD', 'MA']:
        data[place]['data'].loc['2021-04-04', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-04-04', 'death'] = onp.nan
    
    # April 4: missing or anomalous low counts both Sat/Sun
    for place in ['ID', 'CA', 'NM', 'OK']:
        data[place]['data'].loc['2021-04-03', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-04-04', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-04-03', 'death'] = onp.nan
        data[place]['data'].loc['2021-04-04', 'death'] = onp.nan


    # April 4 -- no data Fri/Sat/Sun
    for place in ['NC', 'TN']:
        data[place]['data'].loc['2021-04-02':'2021-04-04', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-04-02':'2021-04-04', 'death'] = onp.nan     
     

    # MI doesn't report on Sundays
    data =  util.load_data()
        # MI doesn't report on Sundays
    if args.run:
        util.run_place(data,
                       args.place,
                       start=args.start,
                       end=args.end,
                       prefix=args.prefix,
                       model_type=config['model'],
                       **config['args'])
    
    util.gen_forecasts(data,
                       args.place,
                       start=args.start,
                       prefix=args.prefix,
                       model_type=config['model'],
                       show=False)
