import covid.models.SEIRD
import covid.models.SEIRD_variable_detection
import covid.models.SEIRD_incident



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

less_rw_frozen = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'rw_scale': 1e-1,
        'num_frozen': 5
    }
}

less_rw_last_5 = {
    'model': covid.models.SEIRD_incident.SEIRD,
    'args'  : {
        'gamma_shape':  100,
        'sigma_shape':  100,
        'rw_scale': 1e-1,
        'rw_use_last': 5
    }
}
