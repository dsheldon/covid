import sys

from . import jhu
from . import covidtracking
from . import states

from covid.models.SEIRD import SEIRD_stochastic

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

def combine_samples(samples, fields=['x', 'y', 'z', 'mean_y', 'mean_z']):
    '''Combine x0, x, x_future and similar fields into a single array'''

    samples = samples.copy()
    
    for f in fields:
        if f in samples:
            f0, f_future = f + '0', f + '_future'
            data = np.concatenate((samples[f0][:,None], samples[f]), axis=1)
            del samples[f0]
            
            if f_future in samples:
                data = np.concatenate((data, samples[f_future]), axis=1)
                del samples[f_future]
            
            samples[f] = data
    
    return samples


def plot_samples(samples, 
                 plot_fields=['I', 'y'], 
                 T=None, 
                 t=None, 
                 ax=None, 
                 n_samples=0,
                 legend=True,
                 model='SEIR'):
    '''
    Plotting method for SIR-type models. 
    (Needs some refactoring to handle both SIR and SEIR)
    '''

    samples = combine_samples(samples)
    n_samples = np.minimum(n_samples, samples['x'].shape[0])
    
    T_data = samples['x'].shape[1]
    if T is None or T > T_data:
        T = T_data
    
    if model == 'SIR':
        S, I, R, C = 0, 1, 2, 3
    elif model == 'SEIR':
        S, E, I, R, H, D, C = 0, 1, 2, 3, 4, 5, 6
    else:
        raise ValueError("Bad model")
    
    fields = {'susceptible'     : samples['x'][:,:T, S],
              'infectious'      : samples['x'][:,:T, I],
              'removed'         : samples['x'][:,:T, R],
              'total infectious': samples['x'][:,:T, C],
              'total confirmed' : samples['y'][:,:T],
              'total deaths'    : samples['z'][:,:T],
              'daily confirmed' : onp.diff(samples['mean_y'][:,:T], axis=1, prepend=np.nan),
              'daily deaths'    : onp.diff(samples['mean_z'][:,:T], axis=1, prepend=np.nan)}
                  
    fields = {k: fields[k] for k in plot_fields}

    medians = {k: np.median(v, axis=0) for k, v in fields.items()}    
    pred_intervals = {k: np.percentile(v, (10, 90), axis=0) for k, v in fields.items()}
    
    t = np.arange(T) if t is None else t[:T]
    
    ax = ax if ax is not None else plt.gca()
    
    df = pd.DataFrame(index=t, data=medians)
    df.plot(ax=ax, legend=legend)
    median_max = df.max().values
    
    colors = [l.get_color() for l in ax.get_lines()]
    
    # Add individual field lines
    if n_samples > 0:
        i = 0
        for k, data in fields.items():
            step = np.array(data.shape[0]/n_samples).astype('int32')
            df = pd.DataFrame(index=t, data=data[::step,:].T)
            df.plot(ax=ax, lw=0.25, color=colors[i], alpha=0.25, legend=False)
            i += 1
    
    # Add prediction intervals
    pi_max = 10
    i = 0
    for k, pred_interval in pred_intervals.items():
        ax.fill_between(t, pred_interval[0,:], pred_interval[1,:], color=colors[i], alpha=0.1, label='CI')
        pi_max = np.maximum(pi_max, np.nanmax(pred_interval[1,:]))
        i+= 1
    
    return median_max, pi_max
    
    
def plot_forecast(post_pred_samples, T, confirmed, 
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
              confirmed_min = 10,
              death_min = 5,
              save = True,
              num_warmup = 1000,
              num_samples = 1000,
              num_chains = 1,
              num_prior_samples = 1000,
              T_future=26*7,
              save_path = 'out',
              **kwargs):

    prob_model = SEIRD_stochastic
    
    print(f"******* {place} *********")
    
    confirmed = data[place]['data'].confirmed[start:]
    death = data[place]['data'].death[start:]
    start = confirmed.index.min()

    confirmed[confirmed < confirmed_min] = np.nan
    death[death < death_min] = np.nan
    
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
                  load_path = 'out',
                  save_path = 'vis',
                  save = True,
                  show = True, 
                  **kwargs):
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    confirmed = data[place]['data'].confirmed[start:]
    death = data[place]['data'].death[start:]
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
        
