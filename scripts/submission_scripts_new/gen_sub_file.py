import covid.util as util
import covid.models.SEIRD_incident as model_type
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import covid.jhu as jhu

import numpy as np
from epiweeks import Week, Year

def construct_daily_df(forecast_start,forecast_samples,target):
     if  target == "dy" or target == "dz":
         t = pd.date_range(start=forecast_start, periods=forecast_samples.shape[1], freq='D')
         daily_df = pd.DataFrame(index=t, data=np.transpose(forecast_samples))
     else:
         t = pd.date_range(start=forecast_start+   pd.Timedelta("1d"), periods=forecast_samples.shape[1]-1, freq='D')
         daily_df = pd.DataFrame(index=t, data=np.transpose(forecast_samples[:,:-1]))

     return daily_df
 
def resample_to_weekly(daily_df,target):
     if target == "dy" or target == "dz":
         weekly_df = daily_df.resample("1w",closed='left',label='left').sum()

     else:
          weekly_df = daily_df.resample("1w",label='left',closed='left').last()#
     weekly_df[weekly_df<0.] = 0.

     return weekly_df 


def get_fips_df(region):
    if region == "US":
       return pd.read_csv('../resources/state_fips_codes.csv')
    elif region=="county":
       county_info = jhu.get_county_info()
       return county_info
    elif region=="EU":
       fips_codes = pd.read_csv('https://raw.githubusercontent.com/epiforecasts/covid19-forecast-hub-europe/main/data-locations/locations_eu.csv')
       return fips_codes

def get_places(region):
   if region == "US":
    
     data = util.load_data()
     state_data = util.load_state_data()
     places = ['US']+ sorted(list(state_data.keys()))

   elif region == "county":
     
     N=500
     with open("../counties.txt") as myfile:
        places = [next(myfile).rstrip('\n') for x in range(N)]

     ignore = ["IN-Marion","IN-Lake","IN-St. Joseph","UT-Weber","IN-Elkhart","DE-Kent","UT-Washington","IN-Porter","MT-Yellowstone","IN-Johnson","IN-Monroe","IN-Madison"]
     places=  [i for i in places if i not in ignore]

      

   elif region == "EU":
     places = STATES_AND_US=["Belgium","Bulgaria","Czechia","Denmark","Germany","Estonia","Ireland","Greece","Spain","France","Croatia","Italy","Cyprus","Latvia","Lithuania","Luxembourg","Hungary","Malta","Netherlands","Austria","Poland","Portugal","Romania","Slovenia","Slovakia","Finland","Sweden","United Kingdom","Iceland","Liechtenstein","Norway","Switzerland"]#['US']+ 

   return places 



def merge_df(fips_codes,forecast,region):
     if region == "US":
          df_truth = forecast.merge(fips_codes, left_on='location', right_on='state', how='left')
          df_truth["state_code"] = df_truth["state_code"].astype(str)
          df_truth = df_truth[["quantile", "value", "type", "state_code","target","forecast_date","target_end_date"]]

          df_truth = df_truth.rename(columns={"state_code": "location"})
     elif region =="county":
          df_truth = forecast.merge(fips_codes, left_on='location', right_on='key', how='left')
          df_truth = df_truth[["quantile", "value", "type", "FIPS","target","forecast_date","target_end_date"]]
          df_truth = df_truth.rename(columns={"FIPS": "location"})
     elif region =="EU":
         
          df_truth = forecast.merge(fips_codes, left_on='location', right_on='location_name', how='left')
          df_truth["location"] = df_truth["location_y"].astype(str)
          df_truth = df_truth[["quantile", "value", "type", "location","target","forecast_date","target_end_date"]]

     return df_truth



## GET SYS ARGUMENTS





forecast_date = "2021-02-07" #pd.to_datetime(sys.argv[1])
model= "llonger_H"#"llonger_H_fix"#sys.argv[2]
samples_directory = "/mnt/nfs/work1/sheldon/sheldon/covid-results/"#sys.argv[3]
#samples_directory="/mnt/nfs/work1/sheldon/gcgibson/"
samples_directory = samples_directory+"/"+model + "/" +forecast_date+"/samples/"


region ="US"#sys.argv[4]
do_z = "T"#sys.argv[5]
do_dz = "T" # sys.argv[6]
do_y = "F" # sys.argv[7]
do_dy = "F" # sys.argv[8]

model_str = "MechBayes" # sys.argv[8]

forecast_date = pd.to_datetime(forecast_date)


if region == "county":
    num_weeks = 8
else:
    num_weeks = 4


# define quantiles

def generate_forecast_df(forecast_date,region,target, target_str,target_type_str,forecast_region,num_weeks,samples_directory):

     if region !="county":
          allQuantiles = [0.01,0.025]+list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99]
     else:
          allQuantiles = [0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975]

     
     forecast_start = forecast_date #+ pd.Timedelta("1d")
     
     model = model_type.SEIRD()
     
     
     forecast = {'quantile':[],'target_end_date':[], 'value':[], 'type':[], 'location':[], 'target':[]}
     
     
     for place in get_places(region):
         # read_samples  
         prior_samples, mcmc_samples, post_pred_samples, forecast_samples = util.load_samples(samples_directory +place +".npz")
         forecast_samples = model.get(forecast_samples, target,forecast=True)
         daily_df = construct_daily_df(forecast_start,forecast_samples,target)
      
         weekly_df = resample_to_weekly(daily_df,target)

         for time, samples in weekly_df.iterrows():
             week_ahead = time.week - forecast_date.week + 1
             for q in allQuantiles:
                      prediction = np.percentile(samples,q*100)
                      forecast["quantile"].append("{:.3f}".format(q))
                      forecast["value"].append(prediction)
                      forecast["type"].append("quantile")
                      forecast["location"].append(place)
                      forecast["target"].append("{:d} wk ahead ".format(week_ahead)+target_type_str + " " + target_str)
                      forecast["forecast_date"] = forecast_date
                      next_saturday = pd.Timedelta('6 days')
                      target_end_date_datetime = pd.to_datetime(time) + next_saturday
                      forecast["target_end_date"].append("{:4d}-{:02d}-{:02d}".format(target_end_date_datetime.year,target_end_date_datetime.month,target_end_date_datetime.day))
                      if q==0.50:
                          forecast["quantile"].append("NA")
                          forecast["value"].append(prediction)
                          forecast["type"].append("point")
                          forecast["location"].append(place)
                          forecast["target"].append("{:d} wk ahead ".format(week_ahead)+target_type_str + " " + target_str)
                          forecast["target_end_date"].append("{:4d}-{:02d}-{:02d}".format(target_end_date_datetime.year,target_end_date_datetime.month,target_end_date_datetime.day))
     forecast = pd.DataFrame(forecast)
     forecast.loc[forecast.type=="point"]
     fips_codes = get_fips_df(region)#pd.read_csv('../../resources/state_fips_codes.csv')
     df_truth = merge_df(fips_codes,forecast,region)
     df_truth['location'] = df_truth['location'].apply(lambda x: '{0:0>2}'.format(x))
     return df_truth

targets_to_run = []
if  do_dz == "T":
   targets_to_run += ["dz"]
if  do_z == "T":
   targets_to_run += ["z"]
if  do_y == "T":
   targets_to_run += ["y"]
if  do_dy == "T":
   targets_to_run += ["dy"]

forecast_df = pd.DataFrame()
for target in targets_to_run:
     if target== "dz" or target == "dy":
         target_type_str = "inc"
     else:
         target_type_str = "cum"
     if target == "dz" or target == "z":
         target_str = "death"
     else:
         target_str = "case"
     forecast_df =  forecast_df.append(generate_forecast_df(forecast_date,region,target, target_str,target_type_str,region,num_weeks,samples_directory)) 

directory =  "submission_files/" + region + "/" + model +"/"  
if not os.path.exists(directory):
    os.makedirs(directory)

fname = directory+ forecast_date.strftime('%Y-%m-%d') + "-UMass-" +model_str+".csv"
forecast_df.to_csv(fname, float_format="%.0f",index=False)
