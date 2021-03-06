#!/bin/bash
model=resample_80_last_10
root=/mnt/nfs/work1/sheldon/gcgibson/
forecast_date="2020-08-09"

incident_samples=$root/$model/$forecast_date/samples/
cumulative_samples=$root/$model/$forecast_date/samples/
county_samples=$root/counties/$forecast_date/samples/


python generate_submission_file_cumulative.py $forecast_date $cumulative_samples

python generate_submission_file_incident.py $forecast_date $incident_samples

python generate_submission_file_counties.py $forecast_date $county_samples

echo "Validating cumulative...."

python validate_submission_file_cumulative.py submission_files/cumulative/$forecast_date-UMass-MechBayes.csv $cumulative_samples

echo "Valdiating incident....."

python validate_submission_file_incident.py submission_files/incident/$forecast_date-UMass-MechBayes.csv $incident_samples

echo "Validating county...."

python validate_submission_file_county.py submission_files/county/$forecast_date-UMass-MechBayes.csv $county_samples

echo "Merging"

python merge_submission_files.py $forecast_date
