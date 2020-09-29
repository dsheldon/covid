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

    # MI and PA don't report on Sundays
    for place in ['MI', 'PA']:
        data[place]['data'].loc['2020-09-27', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-09-27', 'death'] = onp.nan

    # RI, CT, GU don't report on Saturdays/Sundays
    for place in ['RI', 'CT', 'GU']:
        data[place]['data'].loc['2020-09-26', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-09-27', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-09-26', 'death'] = onp.nan
        data[place]['data'].loc['2020-09-27', 'death'] = onp.nan


    # MO dept. of health and human services reports 129 excess deaths
    # added to the system ~Mon-Wed 9/21-9/23 and 63 added on 9/26.
    # These jumps don't seem to match what appears in JHU data, so
    # I am redistributing a similar (slightly smaller) number 
    # of deaths from multiple days during the week
    # https://twitter.com/HealthyLivingMo    
    #
    # Update: these caused instability in fitting so I tweaked them 
    # manually and redistributed fewer deaths
    util.redistribute(data['MO']['data'], '2020-09-22', 10, 60, 'death')
    util.redistribute(data['MO']['data'], '2020-09-23', 25, 60, 'death')
    util.redistribute(data['MO']['data'], '2020-09-25', 10, 60, 'death')
    util.redistribute(data['MO']['data'], '2020-09-26', 25, 60, 'death')
    util.redistribute(data['MO']['data'], '2020-09-27', -4, 60, 'death') # to avoid -1 deaths on last observed day

    # Texas large backlogs on 9/21 and 9/22

    # 9/21 - 2,078 older case recently reported by labs were included
    #        in the statewide total but excluded from statewide and
    #        Bexar County new confirmed cases (103).  3 older cases
    #        recently reported by labs were included in the statewide
    #        total but excluded from statewide and Collin County new
    #        confirmed cases (42).  306 older case recently reported
    #        by labs were included in the statewide total but excluded
    #        from statewide and Dallas County new confirmed cases
    #        (465).  298 older cases recently reported by labs were
    #        included in the statewide total but excluded from
    #        statewide and Frio County new confirmed cases (1).  328
    #        older cases recently reported by labs were included in
    #        the statewide total but excluded from statewide and
    #        Harris County new confirmed cases (225).  1 older case
    #        recently reported by labs was included in the statewide
    #        total but excluded from statewide and Houston County new
    #        confirmed cases (2).  125 older cases recently reported
    #        by labs were included in the statewide total but excluded
    #        from statewide and Tarrant County new confirmed cases
    #        (203).

    #
    # 9/22 - 2 older cases recently reported by labs were included in 
    #        the statewide total but excluded from statewide and
    #        Dallas County new confirmed cases (314).  13,622 older
    #        cases recently reported by labs were included in the
    #        statewide total but excluded from statewide and Harris
    #        County new confirmed cases (507).  231 older cases
    #        recently reported by labs were included in the statewide
    #        total but excluded from statewide and Nueces County new
    #        confirmed cases (1).  1 older cases recently reported by
    #        labs was included in the statewide total but excluded
    #        from statewide and San Jacinto County new confirmed cases
    #        (0).

    # As nearly as I can tell the notes above apply to the previous day
    util.redistribute(data['TX']['data'], '2020-09-20', 2078 + 3 + 306 + 298 + 328 + 1 + 125, 90, 'confirmed')
    util.redistribute(data['TX']['data'], '2020-09-21', 13622 + 231 + 1, 90, 'confirmed')

    # 4,563 new cases 25Sep20 due to the cumulative antigen testing https://covid19.ncdhhs.gov/dashboard 
    util.redistribute(data['NC']['data'], '2020-09-25', 4563, 90, 'confirmed')

    # 139 probable deaths added on Sep 15 https://katv.com/news/local/arkansas-gov-asa-hutchinson-to-give-covid-19-briefing-09-15-2020
    util.redistribute(data['AR']['data'], '2020-09-15', 139, 30, 'death')

    # 577 backlog cases on Sep 17 https://directorsblog.health.azdhs.gov/covid-19-antigen-tests/
    # 764 backlog cases on Sep 18 https://twitter.com/AZDHS
    util.redistribute(data['AZ']['data'], '2020-09-17', 577, 90, 'confirmed')
    util.redistribute(data['AZ']['data'], '2020-09-18', 764, 90, 'confirmed')

    # Correct values 9/15 through 9/20 are: 91,304 92,712 94,746 97,279 99,562 101,227 (source: https://www.dhs.wisconsin.gov/covid-19/cases.htm)
    data['WI']['data'].loc['2020-09-15':'2020-09-20', 'confirmed'] = [91304, 92712, 94746, 97279, 99562, 101227]

    # My best reconstruction of MO backlogs reports ~Sep 5-6. Information
    # here (https://twitter.com/HealthyLivingMo) but exact numbers don't
    # match JHU data, so I redistributed as best I could
    util.redistribute(data['MO']['data'], '2020-09-04', 16, 60)
    util.redistribute(data['MO']['data'], '2020-09-05', 56, 60)
    util.redistribute(data['MO']['data'], '2020-09-06', 15, 60)

    # MA changed definition of confirmed case
    #   fixed in JHU data on Sep 22: 
    #   https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data#retrospective-reporting-of-probable-cases-and-deaths
    # util.redistribute(data['MA']['data'], '2020-09-03', -7936, 90, col='confirmed')

    # Redistribute incident deaths
    util.redistribute(data['IL']['data'], '2020-07-07', 225, 30)
    util.redistribute(data['DE']['data'], '2020-07-24', 45, 30)
    util.redistribute(data['MO']['data'], '2020-07-23', 25, 30)
    util.redistribute(data['TX']['data'], '2020-07-27', 550, 30)

    # PA stopped reporting on Sundays on 2020-09-13. This is the first
    # zero count in the data, so is highly influential. Temporarily
    # impute a reasonable value for cases. Leave deaths alone.
    # data['PA']['data'].loc['2020-09-13', 'confirmed'] = 149567

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
