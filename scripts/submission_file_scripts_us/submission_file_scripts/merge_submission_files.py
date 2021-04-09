import pandas as pd
import sys
forecast_date = sys.argv[1]  
csv1=pd.read_csv("../submission_files/incident/"+forecast_date+"-UMass-MechBayes.csv.clean",dtype=str)
csv2=pd.read_csv("../submission_files/cumulative/"+forecast_date+"-UMass-MechBayes.csv.clean",dtype=str)
csv3=pd.read_csv("../submission_files/county/"+forecast_date+"-UMass-MechBayes.csv.clean",dtype=str)
total = csv1.append(csv2).append(csv3)
total.to_csv("../submission_files/total/"+forecast_date+"-UMass-MechBayes.csv",index=False)
