
import covid.util as util
import covid.models.SEIRD_incident as model_type

import pandas as pd
import matplotlib.pyplot as plt
import sys
forecast_date = pd.to_datetime(sys.argv[1])
samples_directory = sys.argv[2]



import numpy as np
from epiweeks import Week, Year

data = util.load_state_data()
places = ['US']+ sorted(list(data.keys()))
#places = ['AK', 'AL']

allQuantiles = [0.01,0.025]+list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99]

forecast_start = forecast_date +   pd.Timedelta("1d")
model = model_type.SEIRD()


forecast = {'quantile':[],'target_end_date':[], 'value':[], 'type':[], 'location':[], 'target':[]}


for place in places:
    prior_samples, mcmc_samples, post_pred_samples, forecast_samples = util.load_samples(samples_directory +place +".npz")
    forecast_samples = model.get(forecast_samples, 'z',forecast=True)
    t = pd.date_range(start=forecast_start, periods=forecast_samples.shape[1]-1, freq='D')
    daily_df = pd.DataFrame(index=t, data=np.transpose(forecast_samples[:,:-1]))
    point_df = daily_df.median(axis=1)
    weekly_df = daily_df.resample("1w",label='left',closed='left').last()#.apply(lambda x : x[-1])
    weekly_df[weekly_df<0.] = 0.
    for time, samples in weekly_df.iterrows():
        week_ahead = time.week - forecast_date.week + 1

        forecast["quantile"].append("NA")
        forecast["value"].append(np.mean(samples))
        forecast["type"].append("point_mean")
        forecast["location"].append(place)
        forecast["target"].append("{:d} wk ahead cum death".format(week_ahead))
        forecast["target_end_date"].append("NA")
        for q in allQuantiles:
                 deathPrediction = np.percentile(samples,q*100)
                 forecast["quantile"].append("{:.3f}".format(q))
                 forecast["value"].append(deathPrediction)
                 forecast["type"].append("quantile")
                 forecast["location"].append(place)
                 forecast["target"].append("{:d} wk ahead cum death".format(week_ahead))
                 forecast["forecast_date"] = forecast_date
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

fips_codes = pd.read_csv('../resources/state_fips_codes.csv')

df_truth = forecast.merge(fips_codes, left_on='location', right_on='state', how='left')
df_truth["state_code"] = df_truth["state_code"].astype(str)
df_truth = df_truth[["quantile", "value", "type", "state_code","target","forecast_date","target_end_date"]]

df_truth = df_truth.rename(columns={"state_code": "location"})

import datetime

df_truth['location'] = df_truth['location'].apply(lambda x: '{0:0>2}'.format(x))

fname = "../submission_files/cumulative/"+ forecast_date.strftime('%Y-%m-%d') + "-UMass-MechBayes.csv"
df_truth.to_csv(fname, float_format="%.0f",index=False)
