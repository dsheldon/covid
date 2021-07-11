import os
import argparse
import pandas as pd
from pathlib import Path
import time


#STATES_AND_US=["US", "AS", "GU", "MP", "PR", "VI", "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

STATES_AND_US=["Belgium","Bulgaria","Czechia","Denmark","Germany","Estonia","Ireland","Greece","Spain","France","Croatia","Italy","Cyprus","Latvia","Lithuania","Luxembourg","Hungary","Malta","Netherlands","Austria","Poland","Portugal","Romania","Slovenia","Slovakia","Finland","Sweden","United Kingdom","Iceland","Liechtenstein","Norway","Switzerland"]#["Germany","France","United Kingdom","Ireland","Italy"]



#DEFAULT_CONFIGS = ['SEIRD_renewal']# 'llonger_H']

DEFAULT_CONFIGS = ['llonger_H_fix']
#DEFAULT_CONFIGS=['casey','llonger_H_fix','SEIRD_renewal']
#DEFAULT_CONFIGS = ['llonger_H_fix','SEIRD_renewal_ablation2']#,'llonger_H_fix']
DEFAULT_CONFIGS = ['llonger_H_fix','SEIRD_renewal_ablation2']
DEFAULT_CONFIGS = ['SEIRD_renewal_exp_growth']
#TODAY = "2020-06-04"# pd.to_datetime("today").strftime('%Y-%m-%d')
TODAY = "2021-07-04"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Launch cluster jobs for MechBayes')

    # configs
    parser.add_argument('--configs', nargs="+", help='configs to use (from configs.py)', default=DEFAULT_CONFIGS)

    # places
    parser.add_argument('--places',  nargs="+", help='places to run', default=STATES_AND_US)
    parser.add_argument('--places_file', help='file with places to run', default=None)
    parser.add_argument('--num_places', help="use this many places only", type=int, default=None)

    # dates
    parser.add_argument('--start', help='start date', default='2020-03-04')
    parser.add_argument('--forecast_dates', nargs="+", help='forecast dates', default=[TODAY])
    parser.add_argument('--num_sundays', help="use the last n sundays as forecast dates", type=int, default=52)
    #52

    # other
    parser.add_argument('--root', help='root directory for output', default='/mnt/nfs/work1/sheldon/gcgibson/')
    parser.add_argument('--logdir', help='log directory', default='log')
    parser.add_argument('--no-run', help="don't run the model (only do vis)", dest='run', action='store_false')
    parser.add_argument('--sleep', help="time to sleep between sbatch calls", type=float, default=0.1)
    parser.set_defaults(run=True)

    # Parse arguments
    args = parser.parse_args()

    root=args.root
    log=args.logdir
    configs=args.configs

    # Get places: use file if specified, else command-line    
    if args.places_file is not None:
        with open(args.places_file) as f:
            places = f.read().splitlines()	
    else:
        places = args.places

    if args.num_places and args.num_places < len(places):
        places = places[:args.num_places]

    # Get dates
    start = args.start
    if args.num_sundays:
        forecast_dates = list(pd.date_range(periods=args.num_sundays, end=TODAY, freq='W').astype(str))        
    else:
        forecast_dates = args.forecast_dates
        
    extra_args = '' if args.run else '--no-run'


    for config in configs:
        for forecast_date in forecast_dates:
            prefix = f'{root}/{config}/{forecast_date}'
            print(f"prefix is {prefix}")
            
            for place in places:
                name = f'{place}-{forecast_date}-{config}'
                logdir = f'{log}/{config}/{forecast_date}'

                Path(logdir).mkdir(parents=True, exist_ok=True)

                print(f"launching {name}")

                cmd = f'''sbatch \
--job-name="{name}" \
--output="{logdir}/{place}.out" \
--error="{logdir}/{place}.err" \
--nodes=1 \
--ntasks=1 \
--mem=1000 \
--partition=defq \
./run_sir.sh "{place}" --start {start} --end {forecast_date} --config {config} --prefix {prefix} {extra_args}'''
                                
                os.system(cmd)
                time.sleep(args.sleep)
