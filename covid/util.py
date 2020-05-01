import sys

from . import jhu
from . import covidtracking
from . import states

import covid.models.SEIRD

import pandas as pd
import matplotlib.pyplot as plt

import numpy as onp

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from pathlib import Path



"""
************************************************************
Data
************************************************************
"""

def load_world_data():
    # world data
    world = jhu.load_world()
    world = world.loc[:,(slice(None), 'tot', slice(None))] # only country totals

    country_names = world.columns.unique(level=0)
    world_pop_data = pd.read_csv('https://s3.amazonaws.com/rawstore.datahub.io/630580e802a621887384f99527b68f59.csv')
    world_pop_data = world_pop_data.set_index("Country")
        
    country_names_valid = set(country_names) & set(world_pop_data.index) 
    world_data = {
        k: {'data' : world[k].tot, 
            'pop' : world_pop_data.loc[k]['Year_2016'],
            'name' : k}
        for k in country_names_valid
    }

    
    return world_data


def load_state_data(source="jhu"):

    # US state data
    if source=="covidtracker":
        US = covidtracking.load_us()
    if source=="jhu":
        US = jhu.load_us()
    
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



"""
************************************************************
Plotting
************************************************************
"""
    
def plot_forecast(post_pred_samples, 
                  forecast_samples,
                  T, 
                  confirmed, 
                  t = None, 
                  scale='log',
                  n_samples= 100,
                  death = None,
                  daily = False,
                  **kwargs):

    t = t if t is not None else np.arange(T)

    fig, axes = plt.subplots(nrows = 2, figsize=(8,12), sharex=True)
    
    
    if daily:
        variables = ['daily confirmed', 'daily deaths']
        w = 7
        min_periods = 1
        observations = [confirmed.diff().rolling(w, min_periods=min_periods, center=True).mean(), 
                        death.diff().rolling(w, min_periods=min_periods, center=True).mean()]
    else:
        variables = ['total confirmed', 'total deaths']
        observations = [confirmed, death]
        
    for variable, observation, ax in zip(variables, observations, axes):
    
        median_max, pi_max = plot_samples(post_pred_samples, T=T, t=t, ax=ax, plot_fields=[variable], **kwargs)
        observation.plot(ax=ax, style='o')
    
        ax.axvline(observation.index.max(), linestyle='--', alpha=0.5)
        ax.grid(axis='y')

        if scale == 'log':
            ax.set_yscale('log')

            # Don't display below 1
            bottom, top = ax.get_ylim()
            bottom = 1 if bottom < 1 else bottom
            ax.set_ylim([bottom, pi_max])
        else:
            top = np.minimum(2*median_max, pi_max)
            ax.set_ylim([0, top])

    return fig, ax

def plot_R0(mcmc_samples, start):

    fig = plt.figure(figsize=(5,3))
    
    # Compute average R0 over time
    gamma = mcmc_samples['gamma'][:,None]
    beta = mcmc_samples['beta']
    t = pd.date_range(start=start, periods=beta.shape[1], freq='D')
    R0 = beta/gamma

    pi = np.percentile(R0, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'R0': np.median(R0, axis=0)})
    df.plot(style='-o')
    plt.fill_between(t, pi[0,:], pi[1,:], alpha=0.1)

    plt.axhline(1, linestyle='--')
    
    #plt.tight_layout()

    return fig



"""
************************************************************
Running
************************************************************
"""

def run_place(data, 
              place, 
              start = '2020-03-04',
              end = None,
              confirmed_min = 10,
              confirmed_ignore_last = 0,
              death_min = 5,
              save = True,
              num_warmup = 1000,
              num_samples = 1000,
              num_chains = 1,
              num_prior_samples = 1000,
              T_future=26*7,
              save_path = 'out',
              **kwargs):

    prob_model = covid.models.SEIRD.SEIRD()
    
    print(f"******* {place} *********")
    confirmed = data[place]['data'].confirmed[start:end]
    death = data[place]['data'].death[start:end]

    # ignore last few confirmed cases reports
    window_start = confirmed.index.max() - pd.Timedelta(confirmed_ignore_last - 1, "d")
    confirmed[window_start:] = np.nan
    
    start = confirmed.index.min()

#     confirmed[confirmed < confirmed_min] = np.nan
#     death[death < death_min] = np.nan
    
    T = len(confirmed)
    N = data[place]['pop']

    args = {
        'N': N,
        'T': T,
        'rw_scale': 2e-1
    }

    args = dict(args, **kwargs)
    
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
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = f'{path}/{place}_samples.npz'
    np.savez(filename, 
             prior_samples = prior_samples,
             mcmc_samples = mcmc_samples, 
             post_pred_samples = post_pred_samples)


def write_summary(place, mcmc, path='out'):
    # Write diagnostics to file
    Path(path).mkdir(parents=True, exist_ok=True)
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
                  end=None,
                  load_path = 'out',
                  save_path = 'vis',
                  save = True,
                  show = True, 
                  **kwargs):
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    confirmed = data[place]['data'].confirmed[start:end]
    death = data[place]['data'].death[start:end]
    start_ = confirmed.index.min()

    T = len(confirmed)
    N = data[place]['pop']

    prior_samples, mcmc_samples, post_pred_samples = load_samples(place, path=load_path)
    
    for scale in ['log', 'lin']:
        for T in [65, 100, 150]:

            t = pd.date_range(start=start_, periods=T, freq='D')

            fig, ax = plot_forecast(post_pred_samples, T, confirmed, 
                                    t = t, 
                                    scale = scale, 
                                    death = death,
                                    **kwargs)
            
            name = data[place]['name']
            plt.suptitle(f'{name} {T} days ')
            plt.tight_layout()

            if save:
                filename = f'{save_path}/{place}_predictive_scale_{scale}_T_{T}.png'
                plt.savefig(filename)
                
            if show:
                plt.show()
            
    fig = plot_R0(mcmc_samples, start_)    
    plt.title(place)
    plt.tight_layout()
    
    if save:
        filename = f'{save_path}/{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()        


def get_world_pop_data():
     return (dict(zip(names, population)))
        
