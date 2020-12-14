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

# temporary fix for NY Dec 14. Less rw
NY_fix_1 = {
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

# temporary fix for NY Dec 14. Fix higher death probability
NY_fix_2 = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0,
        'death_prob_est': 0.02,
        'death_prob_conc': 1000
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


