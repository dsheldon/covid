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

import cachetools

import scipy
import scipy.stats

from .compartment import SEIRModel

from tqdm import tqdm

import warnings


"""
************************************************************
Data
************************************************************
"""

@cachetools.func.ttl_cache(ttl=600)
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

    world_data['US'] = {'pop': 328000000,'data':world['US'].tot,'name':'US'}
      
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


def load_data(us_source="jhu"):
    state_data = load_state_data(source=us_source)
    world_data = load_world_data()
    return dict(world_data, **state_data)


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
    
def plot_R0(mcmc_samples, start, ax=None):

    ax = plt.axes(ax)
    
    # Compute R0 over time
    gamma = mcmc_samples['gamma'][:,None]
    beta = mcmc_samples['beta']
    t = pd.date_range(start=start, periods=beta.shape[1], freq='D')
    R0 = beta/gamma

    pi = np.percentile(R0, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'R0': np.median(R0, axis=0)})
    df.plot(style='-o', ax=ax)
    ax.fill_between(t, pi[0,:], pi[1,:], alpha=0.1)

    ax.axhline(1, linestyle='--')
    

def plot_growth_rate(mcmc_samples, start, model=SEIRModel, ax=None):
    
    ax = plt.axes(ax)

    # Compute growth rate over time
    beta = mcmc_samples['beta']
    sigma = mcmc_samples['sigma'][:,None]
    gamma = mcmc_samples['gamma'][:,None]
    t = pd.date_range(start=start, periods=beta.shape[1], freq='D')

    growth_rate = SEIRModel.growth_rate((beta, sigma, gamma))

    pi = np.percentile(growth_rate, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'growth_rate': np.median(growth_rate, axis=0)})
    df.plot(style='-o', ax=ax)
    ax.fill_between(t, pi[0,:], pi[1,:], alpha=0.1)

    ax.axhline(0, linestyle='--')
    


"""
************************************************************
Running
************************************************************
"""

def run_place(data, 
              place, 
              model_type=covid.models.SEIRD.SEIRD,
              start = '2020-03-04',
              end = None,
              save = True,
              num_warmup = 1000,
              num_samples = 1000,
              num_chains = 1,
              num_prior_samples = 0,              
              T_future=4*7,
              prefix = "results",
              resample=None,
              **kwargs):


    numpyro.enable_x64()

    print(f"Running {place} (start={start}, end={end})")
    place_data = data[place]['data'][start:end]
    T = len(place_data)

    model = model_type(
        data = place_data,
        T = T,
        N = data[place]['pop'],
        **kwargs
    )
    
    print(" * running MCMC")
    mcmc_samples = model.infer(num_warmup=num_warmup, 
                               num_samples=num_samples)

    if resample is not None:
        print(" * resampling")
        mcmc_samples = model.resample(resample=resample, **kwargs)

    # Prior samples
    prior_samples = None
    if num_prior_samples > 0:
        print(" * collecting prior samples")
        prior_samples = model.prior(num_samples=num_prior_samples)

    # In-sample posterior predictive samples (don't condition on observations)
    print(" * collecting predictive samples")
    post_pred_samples = model.predictive()

    # Forecasting posterior predictive (do condition on observations)
    print(" * collecting forecast samples")
    forecast_samples = model.forecast(T_future=T_future)
        
    if save:

        # Save samples
        path = Path(prefix) / 'samples'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.npz'
        
        save_samples(filename,
                     prior_samples,
                     mcmc_samples, 
                     post_pred_samples,
                     forecast_samples)
        
        path = Path(prefix) / 'summary'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.txt'
        
        write_summary(filename, model.mcmc)

        
def save_samples(filename, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples,
                 forecast_samples):
    
    np.savez(filename, 
             prior_samples = prior_samples,
             mcmc_samples = mcmc_samples, 
             post_pred_samples = post_pred_samples,
             forecast_samples = forecast_samples)


def write_summary(filename, mcmc):
    # Write diagnostics to file
    orig_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        mcmc.print_summary()
    sys.stdout = orig_stdout

    
def load_samples(filename):

    x = np.load(filename, allow_pickle=True)
    
    prior_samples = x['prior_samples'].item()
    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()
    
    return prior_samples, mcmc_samples, post_pred_samples, forecast_samples


def gen_forecasts(data, 
                  place, 
                  model_type=covid.models.SEIRD.SEIRD,
                  start = '2020-03-04', 
                  end=None,
                  save = True,
                  show = True, 
                  prefix='results',
                  **kwargs):
    

    # Deal with paths
    samples_path = Path(prefix) / 'samples'
    vis_path = Path(prefix) / 'vis'
    vis_path.mkdir(parents=True, exist_ok=True)
    
    model = model_type()

    confirmed = data[place]['data'].confirmed[start:end]
    death = data[place]['data'].death[start:end]

    T = len(confirmed)
    N = data[place]['pop']

    filename = samples_path / f'{place}.npz'   
    _, mcmc_samples, post_pred_samples, forecast_samples = load_samples(filename)
        
    for daily in [False, True]:
        for scale in ['log', 'lin']:
            for T in [14, 28]:

                fig, axes = plt.subplots(nrows = 2, figsize=(8,12), sharex=True)    

                if daily:
                    variables = ['dy', 'dz']
                    observations = [confirmed.diff(), death.diff()]
                else:
                    variables = ['y', 'z']
                    observations= [confirmed, death]

                for variable, obs, ax in zip(variables, observations, axes):
                    model.plot_forecast(variable,
                                        post_pred_samples,
                                        forecast_samples,
                                        start,
                                        T_future=T,
                                        obs=obs,
                                        ax=ax,
                                        scale=scale)

                name = data[place]['name']
                plt.suptitle(f'{name} {T} days ')
                plt.tight_layout()

                if save:
                    filename = vis_path / f'{place}_scale_{scale}_daily_{daily}_T_{T}.png'
                    plt.savefig(filename)

                if show:
                    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,4))
    plot_growth_rate(mcmc_samples, start, ax=ax)
    plt.title(place)
    plt.tight_layout()
    
    if save:
        filename = vis_path / f'{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()   
        
        
        
"""
************************************************************
Performance metrics
************************************************************
"""

def score_place(forecast_date,
                data,
                place,
                model_type=covid.models.SEIRD.SEIRD,
                prefix="results"):

    '''Gives performance metrics for each time horizon for one place'''
    
    filename = Path(prefix) / 'samples' / f'{place}.npz'
    prior_samples, mcmc_samples, post_pred_samples, forecast_samples = \
        load_samples(filename)

    model = model_type()

    start = pd.to_datetime(forecast_date) + pd.Timedelta("1d")

    # cumulative deaths 
    obs = data[place]['data'][start:].death
    end = obs.index.max()

    # predicted deaths
    z = model.get(forecast_samples, 'z', forecast=True)
    
    # truncate to smaller length
    T = min(len(obs), z.shape[1])
    z = z[:,:T]
    obs = obs.iloc[:T]
    
    # create data frame for analysis
    samples = pd.DataFrame(index=obs.index, data=z.T)

    n_samples = samples.shape[1]

    # Construct output data frame
    scores = pd.DataFrame(index=obs.index)
    scores['place'] = place
    scores['forecast_date'] = pd.to_datetime(forecast_date)
    scores['horizon'] = (scores.index - scores['forecast_date'])/pd.Timedelta("1d")
    
    # Compute MAE
    point_forecast = samples.median(axis=1)
    scores['err'] = obs-point_forecast

    # Compute log-score
    within_100 = samples.sub(obs, axis=0).abs().lt(100)
    prob = (within_100.sum(axis=1)/n_samples)
    log_score = prob.apply(np.log).clip(lower=-10).rename('log score')
    scores['log_score'] = log_score

    # Compute quantile of observed value in samples
    scores['quantile'] = samples.lt(obs, axis=0).sum(axis=1) / n_samples
    
    return scores

def score_forecast(forecast_date,
                   data, 
                   places=None, 
                   model_type=covid.models.SEIRD.SEIRD,
                   prefix="results"):

    
    if places is None:
        places = list(data.keys())
        places = ['US'] + places
    # Assemble performance metrics each place and time horizon
    details = pd.DataFrame()
    
    print(f'Scoring all places for {forecast_date} forecast')
    
    for place in tqdm(places):
                

        try:
            place_df = score_place(forecast_date,
                                   data,
                                   place,
                                   model_type=model_type,
                                   prefix=prefix)
        except Exception as e:
            warnings.warn(f'Could not score {place}')
            print(e)
        else:
            details = details.append(place_df)

        
    # Now summarize over places for each time horizon
    dates = details.index.unique()
    summary = pd.DataFrame(index=dates)
    
    for date in dates:
        
        horizon = int((date-pd.to_datetime(forecast_date))/pd.Timedelta("1d"))
        rows = details.loc[date]
        
        summary.loc[date, 'horizon'] = horizon

        # Compute signer error / bias
        summary.loc[date, 'signed_err'] = rows['err'].mean()
        
        # Compute MAE
        summary.loc[date, 'MAE'] = rows['err'].abs().mean()
        
        # Compute avg. log-score
        summary.loc[date, 'log_score'] = rows['log_score'].mean()
        
        # Compute KS statistic
        ks, pval = scipy.stats.kstest(rows['quantile'], 'uniform')
        summary.loc[date,'KS'] = ks
        summary.loc[date,'KS_pval'] = pval
        
    summary['forecast_date'] = forecast_date
    
    return summary, details
