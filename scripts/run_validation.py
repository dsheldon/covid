import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax.random import PRNGKey
import sys
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import pandas as pd

import covid
import covid.util as util

import covid.models.SEIRD_variable_detection
import covid.models.SEIRD
if __name__ == "__main__": 
    place = sys.argv[1]
    data = util.load_state_data()


    
    util.run_place(data, 
               place, 
               start='2020-03-15', 
               end='2020-04-27',
               T_future=8*7,
               model_abrv = "SEIRD",
               model_type = covid.models.SEIRD.SEIRD,
               num_warmup=100, 
               num_samples=100)

    
    util.run_place(data, 
               place, 
               start='2020-03-15', 
               end='2020-04-27',
               T_future=8*7,
               model_abrv = "SEIRD_variable_detection",
               model_type = covid.models.SEIRD_variable_detection.SEIRD,
               num_warmup=100, 
               num_samples=100)
