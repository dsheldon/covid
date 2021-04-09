
import covid.util as util

import pandas as pd
import matplotlib.pyplot as plt
import sys
start = sys.argv[1]
forecast_start = sys.argv[2]
samples_directory = sys.argv[3]



import numpy as np
from epiweeks import Week, Year

num_weeks = 8
data = util.load_state_data()
places = sorted(list(data.keys()))
#places = ['AK', 'AL']

allQuantiles = [0.01,0.025]+list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99]

forecast_date = pd.to_datetime(forecast_start)
currentEpiWeek = Week.fromdate(forecast_date) 


forecast = {'quantile':[],'target_end_date':[], 'value':[], 'type':[], 'location':[], 'target':[]}


for place in places:
    prior_samples, mcmc_samples, post_pred_samples, forecast_samples = util.load_samples(place, path=samples_directory)
    forecast_samples = forecast_samples['mean_z_future']
    t = pd.date_range(start=forecast_start, periods=forecast_samples.shape[1], freq='D')
    weekly_df = pd.DataFrame(index=t, data=np.transpose(forecast_samples)).resample("1w",label='left').last()
    weekly_df[weekly_df<0.] = 0.
    for time, samples in weekly_df.iterrows():
        for q in allQuantiles:
                 deathPrediction = np.percentile(samples,q*100)
                 forecast["quantile"].append("{:.3f}".format(q))
                 forecast["value"].append(deathPrediction)
                 forecast["type"].append("quantile")
                 forecast["location"].append(place)
                 horizon_date = Week.fromdate(time)
                 week_ahead = horizon_date.week - currentEpiWeek.week + 1
                 forecast["target"].append("{:d} wk ahead cum death".format(week_ahead))
                 currentEpiWeek_datetime = currentEpiWeek.startdate()
                 forecast["forecast_date"] = "{:4d}-{:02d}-{:02d}".format(currentEpiWeek_datetime.year,currentEpiWeek_datetime.month,currentEpiWeek_datetime.day)
                 next_saturday = pd.Timedelta('6 days')
                 target_end_date_datetime = pd.to_datetime(time) + next_saturday
                 forecast["target_end_date"].append("{:4d}-{:02d}-{:02d}".format(target_end_date_datetime.year,target_end_date_datetime.month,target_end_date_datetime.day))                 
                 if q==0.50:
                     forecast["quantile"].append("NA")
                     forecast["value"].append(deathPrediction)
                     forecast["type"].append("point")
                     forecast["location"].append(place)
                     forecast["target"].append("{:d} wk ahead cum death".format(week_ahead))
                     forecast["target_end_date"].append("{:4d}-{:02d}-{:02d}".format(target_end_date_datetime.year,target_end_date_datetime.month,target_end_date_datetime.day))
forecast = pd.DataFrame(forecast)
forecast.loc[forecast.type=="point"]

fips_codes = pd.read_csv('state_fips_codes.csv')

df_truth = forecast.merge(fips_codes, left_on='location', right_on='state', how='left')
df_truth["state_code"] = df_truth["state_code"].astype(int)
df_truth = df_truth[["quantile", "value", "type", "state_code","target","forecast_date","target_end_date"]]

df_truth = df_truth.rename(columns={"state_code": "location"})

import datetime

df_truth['location'] = df_truth['location'].apply(lambda x: '{0:0>2}'.format(x))

fname = "submission_files/"+forecast_start + "-UMass-MechBayes.csv"
df_truth.to_csv(fname, float_format="%.0f",index=False)
