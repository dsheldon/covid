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

    data =  util.load_data()
        # MI doesn't report on Sundays
    #   Oct 19 - add MS
    for place in ['MI', 'NH', 'MS']:
        data[place]['data'].loc['2021-02-14', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-02-14', 'death'] = onp.nan

    # RI, CT, GU don't report on Saturdays/Sundays
    #   Oct 19 -- add WI (removed Oct 25)
    #   Oct 18 -- add KS
    for place in ['RI', 'GU', 'KS', 'WA', 'CT']:
        data[place]['data'].loc['2021-02-13', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-02-14', 'confirmed'] = onp.nan
        data[place]['data'].loc['2021-02-13', 'death'] = onp.nan
        data[place]['data'].loc['2021-02-14', 'death'] = onp.nan


    # https://covidtracking.com/data/state/ohio/notes
    util.redistribute(data['OH']['data'], '2021-02-11', 650, 90, 'death')
    util.redistribute(data['OH']['data'], '2021-02-12', 2500, 90, 'death')
    util.redistribute(data['OH']['data'], '2021-02-13', 1125, 90, 'death')

    # https://content.govdelivery.com/accounts/AKDHSS/bulletins/2be6de2
    # "All 17 deaths were identified through death certificate review"
    util.redistribute(data['AK']['data'], '2021-02-02', 17, 60, 'death')

    # https://twitter.com/Delaware_DHSS/status/1357838111879921671
    #util.redistribute(data['DE']['data'], '2021-02-05', 54, 90, 'death')
    # https://twitter.com/Delaware_DHSS/status/1357060070538940420
    #util.redistribute(data['DE']['data'], '2021-02-03', 17, 30, 'death')

    # See JHU github
    util.redistribute(data['IN']['data'], '2021-02-04', 150, 90, 'death')

    # https://www.kwch.com/2021/02/05/1st-child-death-from-covid-19-reported-in-kansas/
    # A KDHE spokesperson said that the department was reviewing death 
    # certificates, which contributes to the increase in deaths.
    #util.redistribute(data['KS']['data'], '2021-02-02', 150, 60, 'death')

    # can't find a specific record
    util.redistribute(data['MT']['data'], '2021-02-03', 25, 60, 'death')


    # No details released, but pretty sure these are older deaths
    # https://www.kcrg.com/2021/01/31/reported-covid-19-deaths-in-iowa-swell-to-over-4900/
    util.redistribute(data['IA']['data'], '2021-01-31', 240, 60, 'death')
    util.redistribute(data['IA']['data'], '2021-01-30', 30, 60, 'death')

    # https://twitter.com/scdhec/status/1354893314777088008
    util.redistribute(data['SC']['data'], '2021-01-27', 54, 30, 'death')
    util.redistribute(data['SC']['data'], '2021-01-28', 200, 30, 'death')

    # https://health.hawaii.gov/news/covid-19/hawaii-covid-19-daily-news-digest-january-26-2021/
    util.redistribute(data['HI']['data'], '2021-01-26', 59, 90, 'death')


    util.redistribute(data['MT']['data'], '2021-01-23', 30, 60, 'death')

    # https://content.govdelivery.com/accounts/AKDHSS/bulletins/2bb208d
    util.redistribute(data['AK']['data'], '2021-01-23', 5, 60, 'death')

    # https://content.govdelivery.com/accounts/AKDHSS/bulletins/2ba597a
    util.redistribute(data['AK']['data'], '2021-01-20', 22, 60, 'death')


    util.redistribute(data['WI']['data'], '2021-01-16', 60, 20, 'death')

    # Rebalance large pos/neg vaules Jan 7/8
    util.redistribute(data['NE']['data'], '2021-01-08', -90, 1, 'death')

    # 23 of the deaths reported on the 16th were from between Dec 24 and Jan 16
    # https://www.pressherald.com/2021/01/16/maine-cdc-reports-30-deaths-444-new-cases-of-covid-19/
    util.redistribute(data['ME']['data'], '2021-01-16', 23, 24, 'death')

    # 35 of the deaths reported on Jan 8 were from December. 
    # https://www.wabi.tv/2021/01/08/maine-sees-deadliest-day-of-pandemic-with-41-deaths-789-new-cases/
    util.redistribute(data['ME']['data'], '2021-01-08', 35, 40, 'death')


    # The WA saga....
    util.redistribute(data['WA']['data'], '2021-02-08', 20, 60, 'death')
    util.redistribute(data['WA']['data'], '2021-02-09', 20, 60, 'death')
    util.redistribute(data['WA']['data'], '2021-02-10', 10, 60, 'death')
    util.redistribute(data['WA']['data'], '2021-02-12', 10, 60, 'death')

    # and again!
    util.redistribute(data['WA']['data'], '2021-01-22', 20, 6, 'death')
    util.redistribute(data['WA']['data'], '2021-01-21', 100, 5, 'death')
    util.redistribute(data['WA']['data'], '2021-01-19', 25, 3, 'death')

    # WA weirdness seems to be weekly....
    util.redistribute(data['WA']['data'], '2021-01-12', 60, 3, 'death')
    util.redistribute(data['WA']['data'], '2021-01-13', 20, 4, 'death')

    # More WA cleanup after New Year's. Sigh.
    # Used time series download from WA dashboard as reference, but could not
    # make numbers match closely. Dashboard reports ~20 or fewer deaths each
    # day since start of Jan
    util.redistribute(data['WA']['data'], '2021-01-03', 5000, 2, 'confirmed')
    util.redistribute(data['WA']['data'], '2021-01-08', 30, 30, 'death')
    util.redistribute(data['WA']['data'], '2021-01-08', 10, 7, 'death')
    util.redistribute(data['WA']['data'], '2021-01-07', 10, 30, 'death')
    util.redistribute(data['WA']['data'], '2021-01-06', 30, 30, 'death')
    util.redistribute(data['WA']['data'], '2021-01-06', 10, 5, 'death')
    util.redistribute(data['WA']['data'], '2021-01-05', 25, 30, 'death')
    util.redistribute(data['WA']['data'], '2021-01-05', 10, 4, 'death')
    util.redistribute(data['WA']['data'], '2021-01-04', 15, 3, 'death')


    # Manual smoothing due to combined lack of reporting after Christmas and 
    # imprecise report that a backlog of "approximately 200 deaths" were 
    # reported ~12-29. 
    # https://covid-tracking-project-data.s3.us-east-1.amazonaws.com/state_screenshots/WA/WA-20201230-001452.png
    util.redistribute(data['WA']['data'], '2020-12-29', 120, 60, 'death')
    util.redistribute(data['WA']['data'], '2020-12-29', 40, 4, 'death')
    util.redistribute(data['WA']['data'], '2020-12-30', 20, 60, 'death')
    util.redistribute(data['WA']['data'], '2020-12-31', 20, 60, 'death')


    # 2020-12-20
    # manual smoothing of WA after data update left things very wonky
    util.redistribute(data['WA']['data'], '2020-12-16', -1600, -3, 'confirmed')
    util.redistribute(data['WA']['data'], '2020-12-16', 80, 7, 'death')
    util.redistribute(data['WA']['data'], '2020-12-17', 25, 7, 'death')
    util.redistribute(data['WA']['data'], '2020-12-17', 20, -1, 'death')
    util.redistribute(data['WA']['data'], '2020-12-17', 20, -2, 'death')

    # 2020-12-20
    # California dashboard included 15,337 historical cases in their December 16 update
    # https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data
    util.redistribute(data['CA']['data'], '2020-12-16', 15337, 60, 'confirmed')


    # 2020-12-07: manual smoothing of MA/ME data on Thanksgiving and following
    util.redistribute(data['MA']['data'], '2020-11-26', -3000, -7, 'confirmed')
    util.redistribute(data['MA']['data'], '2020-11-27', 500, 1, 'confirmed')
    util.redistribute(data['MA']['data'], '2020-11-30', -1500, -6, 'confirmed')
    util.redistribute(data['MA']['data'], '2020-12-03', 1500, 7, 'confirmed')

    util.redistribute(data['ME']['data'], '2020-11-27', -200, -7, 'confirmed')
    util.redistribute(data['ME']['data'], '2020-11-28', -200, -7, 'confirmed')
    util.redistribute(data['ME']['data'], '2020-12-03', 60, 7, 'confirmed')
    

    # 1922 antigen tests first reported on Dec 9th. 
    # https://www.health.nd.gov/news/positive-covid-19-test-results-249
    util.redistribute(data['ND']['data'], '2020-12-09', 1922, 60, 'confirmed')

    # Iowa deaths messed up due to change in reporting. Pieced together
    # by using Covid Tracking and news reports
    #
    # https://twitter.com/natalie_krebs?lang=en
    # https://www.iowapublicradio.org/health/2020-12-08/iowa-officials-announce-change-in-methodology-that-raises-covid-19-death-count-by-175
    data['IA']['data'].loc['2020-12-7':'2020-12-12','death'] = [2919, 3021, 3120, 3197, 3212, 3213]
    util.redistribute(data['IA']['data'], '2020-12-07', 175, 60, 'death')

    # AL antigen backlogs in December
    # (https://alpublichealth.maps.arcgis.com/apps/opsdashboard/index.html#/6d2771faa9da4a2786a509d82c8cf0f7)
    util.redistribute(data['AL']['data'], '2020-12-02', 706, 60, 'confirmed')
    util.redistribute(data['AL']['data'], '2020-12-08', 1038 + 473, 60, 'confirmed')
    util.redistribute(data['AL']['data'], '2020-12-10', 473, 60, 'confirmed')
    util.redistribute(data['AL']['data'], '2020-12-12', 398, 60, 'confirmed')

    #  13000 case backlog (JHU CSSE)
    util.redistribute(data['OH']['data'], '2020-12-08', 13000, 60, 'confirmed')

    # JHU redistribution error for WI
    util.redistribute(data['WI']['data'], '2020-10-19', 11000, 3, 'confirmed')

    # GA backlog on Nov 3 (JHU CSSE)
    util.redistribute(data['GA']['data'], '2020-11-03', 29937, 60, 'confirmed')
    util.redistribute(data['GA']['data'], '2020-11-03', 450, 60, 'death')


    # Backlogs from LA county on 10/22, 10/23, 10/24
    #  - https://twitter.com/lapublichealth
    util.redistribute(data['CA']['data'], '2020-10-22', 2000, 60, 'confirmed')
    util.redistribute(data['CA']['data'], '2020-10-23', 2000, 60, 'confirmed')
    util.redistribute(data['CA']['data'], '2020-10-24', 1200, 60, 'confirmed')

    # AL backlogs of cases on 10/23 and 10/24
    #  https://github.com/CSSEGISandData/COVID-19/issues/3264
    #  - 2565 on 10/22 (appar in JHU on 10/23) - from June through Oct 18
    #  - "majority of" 1182 on 10/23 (appear in JHU on 10/24) - from April through Sep
    #    util.redistribute(data['AL']['data'], '2020-10-23', 2565, 100, 'confirmed')
    #    util.redistribute(data['AL']['data'], '2020-10-24', 1182, 100, 'confirmed')


    # NH: 129 old cases on 2020-10-02 
    # https://www.nh.gov/covid19/news/documents/covid-19-update-10022020.pdf
    # util.redistribute(data['NH']['data'], '2020-10-02', 139, 90, 'confirmed')
    # #   some gaps in JHU filled with covidtracking
    # data['NH']['data'].loc['2020-09-17', 'confirmed'] = 7814
    # data['NH']['data'].loc['2020-10-05', 'confirmed'] = 8680
    # data['NH']['data'].loc['2020-10-07', 'confirmed'] = 8800

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
    # util.redistribute(data['MO']['data'], '2020-09-22', 20, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-09-23', 65, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-09-25', 30, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-09-26', 55, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-09-27', -4, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-02', 60, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-03', 20, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-09', 100, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-15', -100, 2, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-17', 100, 30, 'death')
    # util.redistribute(data['MO']['data'], '2020-10-24', 90, 30, 'death')


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

    def fill_nan(A):
             '''
             interpolate to fill nan values
             '''
             inds = onp.arange(A.shape[0])
             good = onp.where(onp.isfinite(A))
             f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
             B  = onp.where(onp.isfinite(A),A,f(inds))
             return B

    from scipy import interpolate
    if args.config =="casey" or args.config=="SEIRD_renewal":
        for place_ in ['Sweden','Switzerland','Spain']:
              confirmed_tmp = onp.diff(data[place_]['data']['confirmed'])
              confirmed_tmp_first_20 = confirmed_tmp[:20] 
              confirmed_last_20 = confirmed_tmp[20:]
              confirmed_last_20 = onp.array([onp.nan if x==0 else x for x in confirmed_last_20])
              confirmed_last_20 = fill_nan(confirmed_last_20)
              confirmed_last_20[-1] = confirmed_last_20[-3]
              confirmed_last_20[-2] = confirmed_last_20[-3] 
              confirmed_last_20 = onp.nan_to_num(confirmed_last_20)#onp.array([0 if x== onp.nan else x for x in confirmed_last_20])
              data[place_]['data']['confirmed'] =onp.append(onp.array([0]),onp.cumsum(onp.append(confirmed_tmp_first_20,confirmed_last_20)))            
    
              confirmed_tmp = onp.diff(data[place_]['data']['death'])
              confirmed_tmp_first_20 = confirmed_tmp[:20]
              confirmed_last_20 = confirmed_tmp[20:]
              confirmed_last_20 = onp.array([onp.nan if x==0 else x for x in confirmed_last_20])
              confirmed_last_20 = fill_nan(confirmed_last_20)
              confirmed_last_20[-1] = confirmed_last_20[-3]
              confirmed_last_20[-2] = confirmed_last_20[-3]
              confirmed_last_20 = onp.nan_to_num(confirmed_last_20)#onp.array([0 if x== onp.nan else x for x in confirmed_last_20])
              data[place_]['data']['death'] = onp.append(onp.array([0]),onp.cumsum(onp.append(confirmed_tmp_first_20,confirmed_last_20))).astype(int)
    elif args.config=="blarb":
    # MI doesn't report on Sundays
    #   Oct 19 - add MS
        for place_ in ['Sweden','Luxembourg','Switzerland','Spain']:
            for date in list(pd.date_range(start=args.start, end=args.end, freq='W').astype(str)):    
                data[place_]['data'].loc[date, 'confirmed'] = onp.nan
                data[place_]['data'].loc[date, 'death'] = onp.nan
            for date in list((pd.date_range(start=args.start, end=pd.to_datetime(args.end), freq='W')-pd.Timedelta("1 day")).astype(str)):
               data[place_]['data'].loc[date, 'confirmed'] = onp.nan
               data[place_]['data'].loc[date, 'death'] = onp.nan



    util.redistribute(data['France']['data'], '2021-03-28', 500, 90, 'death')
    util.redistribute(data['France']['data'], '2021-03-28',2000,90,'confirmed')
    # Correct values 9/15 through 9/20 are: 91,304 92,712 94,746 97,279 99,562 101,227 (source: https://www.dhs.wisconsin.gov/covid-19/cases.htm)

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
