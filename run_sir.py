import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import pandas as pd

from models import SIR_stochastic, SEIR_stochastic, plot_samples

import jhu
import covidtracking
import states
import sys

def load_data():

    # world data
    world = jhu.load_world()
    world = world.loc[:,(slice(None), 'tot', slice(None))] # only country totals
    country_names = world.columns.unique(level=0)
    
    # Need country populations!
    pop = {
        'Italy': 60.48e6,
        'US': 3.27e8,
    }

    data = {country: world[country].tot for country in country_names}

    place_names = {country: country for country in country_names}
    place_names['US'] = 'United States'
    place_names = dict(place_names, **states.states)
    
    # US state data
    US = covidtracking.load_us()
    traits = states.uga_traits()

    state_pop = { k: traits.totalpop[k] for k in traits.index }
    state_data = { k: US[k] for k in US.columns.unique(level=0) }

    # combine them
    data = dict(data, **state_data)
    pop = dict(pop, **state_pop)
    
    return data, pop, place_names


def save_samples(place, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples):
    
    # Save samples
    filename = f'out/{place}_samples.npz'
    np.savez(filename, 
             prior_samples = prior_samples,
             mcmc_samples = mcmc_samples, 
             post_pred_samples = post_pred_samples)

def save_summary(place, mcmc):
    # Write diagnostics to file
    filename = f'out/{place}_summary.txt'
    orig_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        mcmc.print_summary()
    sys.stdout = orig_stdout

    
def run_place(place):
    
    data, pop, place_names = load_data()

    model = 'SEIR'
    prob_model = SEIR_stochastic

    print(f"******* {place} *********")
    
    start = pd.Timestamp('2020-03-04')
    confirmed = data[place].confirmed[start:]
    start = confirmed.index.min()

    T = len(confirmed)
    N = pop[place]

    args = {
        'N': N,
        'T': T,
        'drift_scale': 2e-1,
        'det_conc': 100,
    }

    print("NUTS")
    kernel = NUTS(prob_model,
                  init_strategy = numpyro.infer.util.init_to_median())

    print("MCMC")
    mcmc = MCMC(kernel, 
                num_warmup=1000, 
                num_samples=2000, 
                num_chains=1)

    print("run")
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

    save_summary(place, mcmc)

    save_samples(place,
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples)
    
    
if __name__ == "__main__": 
    place = sys.argv[1]
    run_place(place)
