import pandas as pd
import sys
forecast_date = sys.argv[1]  
csv1=pd.read_csv("../submission_files/incident/"+forecast_date+"-UMass-SemiMech.csv",dtype=str)
csv2=pd.read_csv("../submission_files/incident/cases/"+forecast_date+"-UMass-SemiMech.csv",dtype=str)
total = csv1.append(csv2)
total.to_csv("../submission_files/total/"+forecast_date+"-UMass-SemiMech.csv",index=False)
