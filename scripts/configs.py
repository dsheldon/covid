import covid.models.SEIRD
import covid.models.SEIRD_variable_detection
import covid.models.SEIRD_incident

import covid.util as util


# 2020-04-25 forecast (?)
SEIRD = {
    'model' : covid.models.SEIRD.SEIRD,
    'args'  : {}                # use defaults
}


# 2020-05-03 forecast
strongest_prior = {
    'model' : covid.models.SEIRD_variable_detection.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100
    }
}


# 2020-05-10 forecast
fit_dispersion = {
    'model' : covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100
    }
}

# State forecasts starting 2020-05-17, all US forecasts 
resample_80_last_10 = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10
    }
}

# State and US forecasts starting 2020-09-06
longer_H = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 18.0
    }
}

# State and US forecasts starting 2020-09-20, except 2020-10-20
llonger_H = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0
    }
}

# temporary fix for Dec 20. Less rw
llonger_H_fix = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'H_duration_est': 25.0
    }
}


# Second temporary fix for Dec 20. 
#   -- changed beta_shape to 10 (default was 1)
#   -- also changed initial distributions for E,I,H,D1 to [0, 1e-4*N]
#   -- also changed mxstep from 500 to 4000 in compartment.py
llonger_H_fix_2 = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0,
        'beta_shape': 10
    }
}


# State and US forecasts 2020-10-20
lllonger_H = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 35.0
    }
}



counties = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'T_future': 8*7
    }
}


