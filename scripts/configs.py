import covid.models.SEIRD
import covid.models.SEIRD_variable_detection
import covid.models.SEIRD_incident

SEIRD = {
    'model' : covid.models.SEIRD.SEIRD,
    'args'  : {}                # use defaults
}

SEIRD_variable_detection = {
    'model' : covid.models.SEIRD_variable_detection.SEIRD,
    'args'  : {}                # use defaults
}

SEIRD_incident = {
    'model' : covid.models.SEIRD_incident.SEIRD,
    'args'  : {}                # use defaults
}
