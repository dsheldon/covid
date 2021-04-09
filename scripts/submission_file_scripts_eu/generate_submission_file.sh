#!/bin/bash
model=casey
root=/mnt/nfs/work1/sheldon/gcgibson/ #/home/gcgibson/covid_eu/covid/scripts/results1/
forecast_date="2021-04-04"
model_name=UMass-SemiMech
incident_samples=$root/$model/$forecast_date/samples/



python generate_submission_file_incident.py $forecast_date $incident_samples $model_name


python generate_submission_file_incident_cases.py $forecast_date $incident_samples $model_name


echo "Valdiating incident....."

#python validate_submission_file_incident.py submission_files/incident/$forecast_date-UMass-MechBayes.csv $incident_samples

