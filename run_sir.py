import sys

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

import pandas as pd
from models import SEIR_stochastic

import util

def run_place(place):
    
    world_data = util.load_world_data()
    state_data = util.load_state_data()
    data = dict(world_data, **state_data)  # all data

    model = 'SEIR'
    prob_model = SEIR_stochastic

    print(f"******* {place} *********")
    
    start = pd.Timestamp('2020-03-04')
    confirmed = data[place]['data'].confirmed[start:]
    start = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    args = {
        'N': N,
        'T': T,
        'rw_scale': 2e-1,
        'det_conc': 10,
    }

    kernel = NUTS(prob_model,
                  init_strategy = numpyro.infer.util.init_to_median())

    mcmc = MCMC(kernel, 
                num_warmup=1000, 
                num_samples=2000, 
                num_chains=1)

    print("starting MCMC")
    mcmc.run(jax.random.PRNGKey(2), 
             obs = confirmed.values,
             **args)

    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()

    # Prior samples for comparison
    prior = Predictive(prob_model, posterior_samples = {}, num_samples = 1000)
    prior_samples = prior(PRNGKey(2), **args)
    
    # Posterior predictive samples for visualization
    args['drift_scale'] = 0 # set drift to zero for forecasting
    post_pred = Predictive(prob_model, posterior_samples = mcmc_samples)
    post_pred_samples = post_pred(PRNGKey(2), T_future=100, **args)

    util.write_summary(place, mcmc)

    util.save_samples(place,
                     prior_samples,
                     mcmc_samples, 
                     post_pred_samples)
    
    
if __name__ == "__main__": 
    place = sys.argv[1]
    run_place(place)
