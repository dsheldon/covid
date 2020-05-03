import covid.util as util
import covid.models.SEIRD_variable_detection
import numpy as np
data = util.load_data()
places = ['KS', 'SC', 'IL', 'WI', 'LA', 'NE', 'FL', 'DE', 'ID', 'RI', 'CA', 'SD', 'MA', 'AK', 'MT', 'MD', 'KY', 'NH', 'NJ', 'AZ', 'GA', 'MO', 'NV', 'WA', 'CO', 'NC', 'TX', 'AL', 'DC', 'CT', 'OK', 'MS', 'TN', 'WY', 'IN', 'UT', 'VA', 'MN', 'ND', 'ME', 'OH', 'HI', 'NY', 'VT', 'AR', 'PA', 'IA', 'WV', 'NM', 'OR', 'MI']
start_eval = '2020-04-27'

average_m1 = []
average_m2 = []
for place in places:
     print (place)
     _,m1_mae = util.score_forecats(start=start_eval,place=place,data=data)

     print (m1_mae)
     average_m1.append(m1_mae)
     err_plotm2,m2_mae = util.score_forecats(start=start_eval,place=place,data=data,
                            model_abrv="SEIRD_variable_detection",
                             model=covid.models.SEIRD_variable_detection.SEIRD())

     print (m2_mae)
     average_m2.append(m2_mae)
     print ("-----")


average_m1 = np.array(average_m1)
average_m2 = np.array(average_m2)

print (np.mean(average_m1))
print (np.mean(average_m2))
