import pandas as pd
import numpy as np
import sys
import covid.util as util
import covid.models.SEIRD_incident as model_type

### get daily incident
sub_file = sys.argv[1]
samples_directory = sys.argv[2]
model = model_type.SEIRD()
prior_samples, mcmc_samples, post_pred_samples, forecast_samples = util.load_samples(samples_directory + "CA-Los Angeles" +".npz")
forecast_samples = model.get(forecast_samples, 'dy',forecast=True)
mean_forecast_samples = np.mean(forecast_samples,axis=0)

data = util.load_county_data()




sub = pd.read_csv("../"+sub_file,dtype=str)


# verify less than 4 weeks

targets = np.unique(sub.target.values)
if (len(targets) > 4):
    print ("Error: more than 4 week ahead present")

else:
    print ("Success: only 4 week ahead or fewer present")
# verify fips codes are length 5
fips = np.unique(sub.location.values)
lengths = [len(str(k)) == 5 for k in fips]
if (all(lengths)==False):
    print ("Error: FIPS incorrect")
else:
    print ("Success: FIPS correctly formatted")

# verify sum of first medians is close to median

print ("Sum of first 7 days LA: " ,np.sum(mean_forecast_samples[:7]))


print ("Sum of 7-14 days LA: " ,np.sum(mean_forecast_samples[7:14]))

print ("Sum of first 14-21 days LA: " ,np.sum(mean_forecast_samples[14:21]))

print ("Sum of first 21-28 days LA: " ,np.sum(mean_forecast_samples[21:28]))

sub_file_success = sub[sub.type != "point_mean"]

sub_file_success.to_csv("../"+sub_file + ".clean", index=False)
