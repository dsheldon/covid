import configs
import covid.util as util
import covid.models.SEIRD_variable_detection
import numpy as np
import pandas as pd


data = util.load_data()

config_names = ['SEIRD', 'SEIRD_variable_detection']
forecast_dates = ["2020-04-04", "2020-04-11",  "2020-04-18", "2020-04-25"]

places = ['KS', 'SC', 'IL', 'WI', 'LA', 'NE', 'FL', 'DE', 'ID', 'RI', 'CA', 'SD', 'MA', 'AK', 'MT', 'MD', 'KY', 'NH', 'NJ', 'AZ', 'GA', 'MO', 'NV', 'WA', 'CO', 'NC', 'TX', 'AL', 'DC', 'CT', 'OK', 'MS', 'TN', 'WY', 'IN', 'UT', 'VA', 'MN', 'ND', 'ME', 'OH', 'HI', 'NY', 'VT', 'AR', 'PA', 'IA', 'WV', 'NM', 'OR', 'MI']

eval_date = '2020-05-02'

print(f"model,forecast_date,place,err")

for config_name in config_names:
     for forecast_date in forecast_dates:

          forecast_start = pd.to_datetime(forecast_date) + pd.Timedelta("1d")
          prefix = f"results/{config_name}/{forecast_date}"

          config = getattr(configs, config_name)          
          errs = []
          
          for place in places:
               
               err = util.score_forecasts(start=forecast_start,
                                          place=place,
                                          data=data,
                                          model_type=config['model'],
                                          eval_date=eval_date,
                                          prefix=prefix)
               
               print(f"{config_name},{forecast_date},{place},{err:.2f}")

               errs.append(err)
          
          mae = np.mean(np.abs(np.array(errs)))
          print(f"{config_name},{forecast_date},MAE,{mae:.2f}")
          
