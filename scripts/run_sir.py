import sys
import argparse
import covid.util as util
import configs
import numpy as onp


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

    data = config.get('data') or util.load_data()

    # Redistribute ~1750 NJ probable deaths added on 2020-06-25
    for place in ['NJ', 'US']:
        util.redistribute(data[place]['data'], '2020-6-25', 1750, 60)

    # Redistribute incident deaths from other places/dates
    util.redistribute(data['IL']['data'], '2020-07-07', 225, 30)
    util.redistribute(data['DE']['data'], '2020-07-24', 45, 30)
    util.redistribute(data['MO']['data'], '2020-07-23', 25, 30)

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
