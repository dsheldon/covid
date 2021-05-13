import pandas as pd
import numpy as np

regions_to_merge = ["US","county"]

total_df = pd.DataFrame()
forecast_date = "2021-05-02" #pd.to_datetime(sys.argv[1])
model_str = "MechBayes" 

for region in regions_to_merge:
    if region == "US":
       model = "llonger_H_fix"
    elif region =="county":
       model = "counties"
    directory =  "submission_files/" + region + "/" + model +"/"
    fname = directory+ forecast_date + "-UMass-" +model_str+".csv"
    tmp = pd.read_csv(fname,dtype=str)
    tmp['location'] = tmp['location'].astype(str)

    total_df = total_df.append(tmp)

directory =  "merged_submission_files/"
fname = directory+ forecast_date + "-UMass-" +model_str+".csv"
total_df.to_csv(fname, float_format="%.2f",index=False)


