import covid.models.SEIRD
import covid.models.SEIRD_variable_detection
import covid.models.casey
import covid.models.SEIRD_renewal
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
#  changed gamma_shape and sigma_shape from 100 to 1000 on 2021-01-10
llonger_H = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0
    }
}


#  changed gamma_shape and sigma_shape from 100 to 1000 on 2021-01-10
casey = {
    'model': covid.models.casey.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'rw_use_last': 10,
        'H_duration_est': 25.0
    }
}


SEIRD_renewal = {
    'model': covid.models.SEIRD_renewal.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'H_duration_est': 25.0
    }
}


# Less rw
llonger_H_fix = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'H_duration_est': 25.0
    }
}

# For debugging on Jan 3
debug = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0,
        'num_warmup': 100,
        'num_samples': 100
    }
}

debug2 = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0
    }
}


# For debugging on Jan 3
fix = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'H_duration_est': 25.0,
        'beta_shape': 1,
        'rw_scale': 1e-1,
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


#  changed gamma_shape and sigma_shape from 100 to 1000 on 2021-01-10
counties = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  1000,
        'sigma_shape':  1000,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'T_future': 8*7
    }
}


counties_fix = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'resample_high': 80,
        'rw_use_last': 10,
        'rw_scale': 1e-1,
        'T_future': 8*7,
        'H_duration_est': 25.0,
        'beta_shape': 1
    }
}
