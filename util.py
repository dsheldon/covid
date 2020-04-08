import jax.numpy as np
import jhu
import covidtracking
import states
import sys

from models import SEIR_stochastic, plot_forecast, plot_R0

import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive




def load_world_data():
    # world data
    world = jhu.load_world()
    world = world.loc[:,(slice(None), 'tot', slice(None))] # only country totals

    country_names = world.columns.unique(level=0)

    world_data = {
        k: {'data' : world[k].tot, 
            'pop' : None,
            'name' : k}
        for k in country_names
    }

    # Need world population data!
    world_data['US']['pop'] = 3.27e8
    world_data['Italy']['pop'] = 60.48e6
    
    return world_data


def load_state_data():

    # US state data
    US = covidtracking.load_us()
    traits = states.uga_traits()

    state_set = set(traits.index) & set(US.columns.unique(level=0))

    state_data = {
        k : {'data': US[k], 
             'pop': traits.totalpop[k],
             'name': traits.NAME[k]
            }
        for k in state_set
    }
    
    return state_data


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
    
    return data, pop, place_names, state_pop


def run_place(data, 
              place, 
              start = '2020-03-04',
              use_hosp = True, 
              save = True,
              num_warmup = 1000,
              num_samples = 1000,
              num_chains = 1,
              num_prior_samples = 1000,
              save_path = 'out'):

    prob_model = SEIR_stochastic
    
    print(f"******* {place} *********")
    
    confirmed = data[place]['data'].confirmed[start:]
    hosp = data[place]['data'].hospitalizedCumulative[start:]
    start = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    args = {
        'N': N,
        'T': T,
        'rw_scale': 2e-1,
        'use_hosp' : use_hosp
    }

    kernel = NUTS(prob_model,
                  init_strategy = numpyro.infer.util.init_to_median())

    mcmc = MCMC(kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples, 
                num_chains=num_chains)

    print("Running MCMC")
    mcmc.run(jax.random.PRNGKey(2),
             obs = confirmed.values,
             hosp = hosp.values,
             **args)

    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    
    # Prior samples for comparison
    prior = Predictive(prob_model, posterior_samples = {}, num_samples = num_prior_samples)
    prior_samples = prior(PRNGKey(2), **args)

    # Posterior predictive samples for visualization
    args['rw_scale'] = 0 # disable random walk for forecasting
    post_pred = Predictive(prob_model, posterior_samples = mcmc_samples)
    post_pred_samples = post_pred(PRNGKey(2), T_future=100, **args)

    if save:
        save_samples(place,
                     prior_samples,
                     mcmc_samples, 
                     post_pred_samples,
                     path=save_path)
        
        write_summary(place, mcmc, path=save_path)

        
def save_samples(place, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples,
                 path='out'):
    
    # Save samples
    filename = f'{path}/{place}_samples.npz'
    np.savez(filename, 
             prior_samples = prior_samples,
             mcmc_samples = mcmc_samples, 
             post_pred_samples = post_pred_samples)


def write_summary(place, mcmc, path='out'):
    # Write diagnostics to file
    filename = f'out/{place}_summary.txt'
    orig_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        mcmc.print_summary()
    sys.stdout = orig_stdout

    
def load_samples(place, path='out'):
    
    filename = f'{path}/{place}_samples.npz'
    x = np.load(filename, allow_pickle=True)
    
    prior_samples = x['prior_samples'].item()
    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    
    return prior_samples, mcmc_samples, post_pred_samples


def gen_forecasts(data, 
                  place, 
                  start = '2020-03-04', 
                  load_path = 'out',
                  save_path = 'vis',
                  save = True,
                  use_hosp = True,
                  show = True):
    
    confirmed = data[place]['data'].confirmed[start:]
    hosp = data[place]['data'].hospitalizedCumulative[start:]
    start_ = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    prior_samples, mcmc_samples, post_pred_samples = load_samples(place, path=load_path)
    
    for scale in ['log', 'lin']:
        for T in [30, 50, 100]:

            t = pd.date_range(start=start_, periods=T, freq='D')

            fig, ax = plot_forecast(post_pred_samples, T, confirmed, 
                                    t = t, 
                                    scale = scale, 
                                    use_hosp = use_hosp, 
                                    hosp = hosp)
            
            name = data[place]['name']
            plt.suptitle(f'{name} {T} days ')

            if save:
                filename = f'{save_path}/{place}_predictive_scale_{scale}_T_{T}.png'
                plt.savefig(filename)
                
            if show:
                plt.show()
            
    fig = plot_R0(mcmc_samples, start_)    
    plt.title(place)
    if save:
        filename = f'{save_path}/{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()