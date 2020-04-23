import jax.numpy as np
import jhu
import covidtracking
import states
import sys

import models

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


def future_data(data, T, offset=1):
    '''Projects data frame with (place, time) MultiIndex into future by
       repeating final time value for each place'''
    data = data.unstack(0)
    orig_start = data.index.min()
    start = data.index.max() + pd.Timedelta(offset, "D")
    future = pd.date_range(start=start, periods=T, freq="D")
    data = data.reindex(future, method='nearest')
    data['t'] = (data.index-orig_start)/pd.Timedelta("1d")
    data = data.stack()
    data.index = data.index.swaplevel(0, 1)
    data = data.sort_index()
    return data



def load_state_Xy(which=None):
    X_place = states.uga_traits().drop('DC') # incomplete data for DC
    X = states.uga_interventions()
    y = covidtracking.load_us_flat()
    
    Xy = y.join(X, how='inner').sort_index()
    
    # Remove dates without enough data
    date = Xy.index.get_level_values(1)
    counts = Xy.groupby(date).apply(lambda x: len(x))
    good_dates = counts.index[counts == counts.max()]
    Xy = Xy.loc[date.isin(good_dates)]
        
    # Add integer time column
    start = Xy.index.unique(level=1).min()
    Xy['t'] = (Xy['date']-start)/pd.Timedelta("1d")
            
    # Select requested states
    if which is not None:
        Xy = Xy.loc[which,:]
        X_place = X_place.loc[which,:]
        
    return Xy, X_place




def run_place(data, 
              place, 
              start = '2020-03-04',
              use_hosp = False, 
              save = True,
              num_warmup = 1000,
              num_samples = 100,
              num_chains = 1,
              num_prior_samples = 1000,
              T_future=26*7,
              save_path = 'out',
              **kwargs):

    prob_model = models.SEIR_stochastic
    
    print(f"******* {place} *********")
    
    confirmed = data[place]['data'].confirmed[start:]
    death = data[place]['data'].death[start:]
    start = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    args = {
        'N': N,
        'T': T,
        'rw_scale': 2e-1,
        'use_hosp' : use_hosp
    }

    args = dict(args, **kwargs)
#     args = {
#         'N': N,
#         'T': T,
#         'rw_scale': 2e-1,
#         'drift_scale': 1e-1, # 5e-2
#         'use_hosp' : use_hosp
#     }

    
    kernel = NUTS(prob_model,
                  init_strategy = numpyro.infer.util.init_to_median())

    mcmc = MCMC(kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples, 
                num_chains=num_chains)

    print("Running MCMC")
    mcmc.run(jax.random.PRNGKey(2),
             obs = confirmed.values,
             death = death.values,
             **args)

    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    
    # Prior samples for comparison
    prior = Predictive(prob_model, posterior_samples = {}, num_samples = num_prior_samples)
    prior_samples = prior(PRNGKey(2), **args)

    # Posterior predictive samples for visualization
    args['rw_scale'] = 0 # disable random walk for forecasting
    post_pred = Predictive(prob_model, posterior_samples = mcmc_samples)
    post_pred_samples = post_pred(PRNGKey(2), T_future=T_future, **args)

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
    death = data[place]['data'].death[start:]
    start_ = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    prior_samples, mcmc_samples, post_pred_samples = load_samples(place, path=load_path)
    
    for scale in ['log', 'lin']:
        for T in [65, 100, 150]:

            t = pd.date_range(start=start_, periods=T, freq='D')

            fig, ax = models.plot_forecast(post_pred_samples, T, confirmed, 
                                    t = t, 
                                    scale = scale, 
                                    use_hosp = use_hosp, 
                                    death = death)
            
            name = data[place]['name']
            plt.suptitle(f'{name} {T} days ')
            plt.tight_layout()

            if save:
                filename = f'{save_path}/{place}_predictive_scale_{scale}_T_{T}.png'
                plt.savefig(filename)
                
            if show:
                plt.show()
            
    fig = models.plot_R0(mcmc_samples, start_)    
    plt.title(place)
    plt.tight_layout()
    
    if save:
        filename = f'{save_path}/{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()