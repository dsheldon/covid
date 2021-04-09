import pandas as pd


regions_to_merge = ["US","county"]

total_df = pd.DataFrame()
forecast_date = "2021-02-07" #pd.to_datetime(sys.argv[1])
model_str = "MechBayes" 

for region in regions_to_merge:
    if region == "US":
       model = "llonger_H"
    elif region =="county":
       model = "counties"
    directory =  "submission_files/" + region + "/" + model +"/"
    fname = directory+ forecast_date + "-UMass-" +model_str+".csv"
    total_df = total_df.append(pd.read_csv(fname))


directory =  "merged_submission_files/"
fname = directory+ forecast_date + "-UMass-" +model_str+".csv"
total_df.to_csv(fname, float_format="%.3f",index=False)


