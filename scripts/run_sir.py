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

    # MI doesn't report on Sundays
    #   Oct 19 - add MS
    for place in ['MI', 'NH', 'MS']:
        data[place]['data'].loc['2020-10-25', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-10-25', 'death'] = onp.nan

    # RI, CT, GU don't report on Saturdays/Sundays
    #   Oct 19 -- add WI (removed Oct 25)
    #   Oct 18 -- add KS
    for place in ['RI', 'CT', 'GU', 'KS']:
        data[place]['data'].loc['2020-10-24', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-10-25', 'confirmed'] = onp.nan
        data[place]['data'].loc['2020-10-24', 'death'] = onp.nan
        data[place]['data'].loc['2020-10-25', 'death'] = onp.nan

    # WA no deaths on weekends: https://www.doh.wa.gov/Emergencies/COVID19/DataDashboard
    for place in ['WA']:
        data[place]['data'].loc['2020-10-24', 'death'] = onp.nan
        data[place]['data'].loc['2020-10-25', 'death'] = onp.nan


    # Backlogs from LA county on 10/22, 10/23, 10/24
    #  - https://twitter.com/lapublichealth
    util.redistribute(data['CA']['data'], '2020-10-22', 2000, 60, 'confirmed')
    util.redistribute(data['CA']['data'], '2020-10-23', 2000, 60, 'confirmed')
    util.redistribute(data['CA']['data'], '2020-10-24', 1200, 60, 'confirmed')

    # AL backlogs of cases on 10/23 and 10/24
    #  https://github.com/CSSEGISandData/COVID-19/issues/3264
    #  - 2565 on 10/22 (appar in JHU on 10/23) - from June through Oct 18
    #  - "majority of" 1182 on 10/23 (appear in JHU on 10/24) - from April through Sep
    util.redistribute(data['AL']['data'], '2020-10-23', 2565, 100, 'confirmed')
    util.redistribute(data['AL']['data'], '2020-10-24', 1182, 100, 'confirmed')


    # NH: 129 old cases on 2020-10-02 
    # https://www.nh.gov/covid19/news/documents/covid-19-update-10022020.pdf
    util.redistribute(data['NH']['data'], '2020-10-02', 139, 90, 'confirmed')
    #   some gaps in JHU filled with covidtracking
    data['NH']['data'].loc['2020-09-17', 'confirmed'] = 7814
    data['NH']['data'].loc['2020-10-05', 'confirmed'] = 8680
    data['NH']['data'].loc['2020-10-07', 'confirmed'] = 8800

    # MO dept. of health and human services reports 129 excess deaths
    # added to the system ~Mon-Wed 9/21-9/23 and 63 added on 9/26.
    # These jumps don't seem to match what appears in JHU data, so
    # I am redistributing a similar (slightly smaller) number 
    # of deaths from multiple days during the week
    # https://twitter.com/HealthyLivingMo
    #
    # More wonky stuff on 10-02 and 10-03. New dashboard has 3 day delay
    # before showing official numbers so *very* hard to tell what should
    # be correct. Given that official MO dashboard shows deaths in single
    # digits most of past 7 days I am assuming the JHU data is due to
    # backlogs or errors
    # 
    # UPDATE: MO deaths is a complete mess. They seem to report backlogs
    # ~once/week. What is below now amounts to just an attempt at smoothing.
    util.redistribute(data['MO']['data'], '2020-09-22', 20, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-09-23', 65, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-09-25', 30, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-09-26', 55, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-09-27', -4, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-10-02', 60, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-10-03', 20, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-10-09', 100, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-10-15', -100, 2, 'death')
    util.redistribute(data['MO']['data'], '2020-10-17', 100, 30, 'death')
    util.redistribute(data['MO']['data'], '2020-10-24', 90, 30, 'death')


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
    #        (203).  Older cases are being reported for several
    #        counties in Public Health Region 8 after DSHS staff
    #        identified 3,921 cases that had not previously been
    #        reported. Those counties are Atascosa (522), Bandera
    #        (41), Calhoun (186), Dimmit (53), Edwards (33), Gillespie
    #        (96), Gonzales (234), Guadalupe (1587), Jackson (77),
    #        Karnes (181), Kendall (128), Kerr (142), Kinney (19),
    #        Lavaca (252), Real (12), Wilson (307) and Zavala
    #        (51). There are no new cases reported for those counties
    #        today.

    # 9/22 - 2 older cases recently reported by labs were included in
    # the statewide total but excluded from statewide and Dallas
    # County new confirmed cases (314).  13,622 older cases recently
    # reported by labs were included in the statewide total but
    # excluded from statewide and Harris County new confirmed cases
    # (507).  231 older cases recently reported by labs were included
    # in the statewide total but excluded from statewide and Nueces
    # County new confirmed cases (1).  1 older cases recently reported
    # by labs was included in the statewide total but excluded from
    # statewide and San Jacinto County new confirmed cases (0).

    # As nearly as I can tell the notes above apply to the previous day
    util.redistribute(data['TX']['data'], '2020-09-20', 2078 + 3 + 306 + 298 + 328 + 1 + 125, 90, 'confirmed')
    util.redistribute(data['TX']['data'], '2020-09-21', 13622 + 231 + 1, 90, 'confirmed')

    # 139 probable deaths added on Sep 15 https://katv.com/news/local/arkansas-gov-asa-hutchinson-to-give-covid-19-briefing-09-15-2020
    util.redistribute(data['AR']['data'], '2020-09-15', 139, 30, 'death')

    # 577 backlog cases on Sep 17 https://directorsblog.health.azdhs.gov/covid-19-antigen-tests/
    # 764 backlog cases on Sep 18 https://twitter.com/AZDHS
    util.redistribute(data['AZ']['data'], '2020-09-17', 577, 90, 'confirmed')
    util.redistribute(data['AZ']['data'], '2020-09-18', 764, 90, 'confirmed')

    # Correct values 9/15 through 9/20 are: 91,304 92,712 94,746 97,279 99,562 101,227 (source: https://www.dhs.wisconsin.gov/covid-19/cases.htm)
    data['WI']['data'].loc['2020-09-15':'2020-09-20', 'confirmed'] = [91304, 92712, 94746, 97279, 99562, 101227]



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
