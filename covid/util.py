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

def load_world_data():
    # world data
    world = jhu.load_world()
    world = world.loc[:,(slice(None), 'tot', slice(None))] # only country totals

    country_names = world.columns.unique(level=0)
    world_pop_data = pd.read_csv('https://s3.amazonaws.com/rawstore.datahub.io/630580e802a621887384f99527b68f59.csv')
    world_pop_data = world_pop_data.set_index("Country")
        
    country_names_valid = set(country_names) & set(world_pop_data.index) 
    
    world_data = {
        k: {'data' : world[k].tot.copy(), 
            'pop' : world_pop_data.loc[k]['Year_2016'],
            'name' : k}
        for k in country_names_valid
    }

    world_data['US'] = {'pop': 328000000,'data':world['US'].tot,'name':'US'}
      
    return world_data

def load_state_data():

    US = jhu.load_us()
    info = jhu.get_state_info()
    
    data = {
        k : {'data': US[k].copy(), 
             'pop': info.loc[k, 'Population'],
             'name': states.states_territories[k]
            }
        for k in info.index
    }
    
    return data

def load_county_data():
    US = jhu.load_us(counties=True)
    info = jhu.get_county_info()
    
    counties = set(info.index) & set(US.columns.unique(level=0))
    
    data = {
        k : {'data': US[k].copy(), 
             'pop': info.loc[k, 'Population'],
             'name': info.loc[k, 'name']
            }
        for k in counties
    }
    
    return data


def load_data():
    state_data = load_state_data()
    world_data = load_world_data()
    county_data = load_county_data()
    return dict(world_data, **state_data, **county_data)


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



def redistribute(df, date, n, k, col='death'):
    '''Redistribute n incident cases/deaths to previous k days'''
    
    # Note: modifies df in place

    # e.g., 100 incident deaths happen on day t
    #   --> n/k incident deaths on days t-k+1, t-k+2, ..., t
    #   --> n/3 incident deaths on days t-2, t-1, 2
    # 
    # the cumulative number by day t does not change
    
    ndays = onp.abs(k)
    
    a = n // ndays
    b = n % ndays
    
    new_incident = a * onp.ones(ndays)
    new_incident[:b] += 1
    
    date = pd.to_datetime(date)
    
    if k > 0:
        new_incident = onp.concatenate([new_incident, [-n]])
        new_cumulative = onp.cumsum(new_incident)    
        end = date 
        start = date - pd.Timedelta('1d') * ndays
    else:
        new_incident = onp.concatenate([[-n], new_incident])
        new_cumulative = onp.cumsum(new_incident)    
        start = date
        end = date + pd.Timedelta('1d') * ndays
    
    days = pd.date_range(start=start, end=end)
    #days = pd.date_range(end=date-pd.Timedelta('1d'), periods=k-1)
    
    df.loc[days, col] += new_cumulative


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

    pi = onp.percentile(R0, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'R0': onp.median(R0, axis=0)})
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

    pi = onp.percentile(growth_rate, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'growth_rate': onp.median(growth_rate, axis=0)})
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
              resample_low=0,
              resample_high=100,
              save_fields=['beta0', 'beta', 'sigma', 'gamma', 'dy0', 'dy', 'dy_future', 'dz0', 'dz', 'dz_future', 'y0', 'y', 'y_future', 'z0', 'z', 'z_future' ],
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

    if resample_low > 0 or resample_high < 100:
        print(" * resampling")
        mcmc_samples = model.resample(low=resample_low, high=resample_high, **kwargs)

    # Prior samples
    prior_samples = None
    if num_prior_samples > 0:
        print(" * collecting prior samples")
        prior_samples = model.prior(num_samples=num_prior_samples)

    # In-sample posterior predictive samples (don't condition on observations)
    print(" * collecting in-sample predictive samples")
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
                     forecast_samples,
                     save_fields=save_fields)
        
        path = Path(prefix) / 'summary'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.txt'
        
        write_summary(filename, model.mcmc)

        
def save_samples(filename, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples,
                 forecast_samples,
                 save_fields=None):
    

    def trim(d):
        if d is not None:
            d = {k : v for k, v in d.items() if k in save_fields}
        return d
        
    onp.savez_compressed(filename, 
                         prior_samples = trim(prior_samples),
                         mcmc_samples = trim(mcmc_samples),
                         post_pred_samples = trim(post_pred_samples),
                         forecast_samples = trim(forecast_samples))


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
            for T in [28, 56]:

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
                prefix="results",
                target='deaths'):
    '''Gives performance metrics for each time horizon for one place'''
    
    if target == 'deaths':
        forecast_field = 'z'
        obs_field = 'death'
    elif target == 'cases':
        forecast_field = 'y'
        obs_field = 'confirmed'
    else:
        raise ValueError('Invalid target')


    filename = Path(prefix) / 'samples' / f'{place}.npz'
    prior_samples, mcmc_samples, post_pred_samples, forecast_samples = \
        load_samples(filename)

    model = model_type()

    start = pd.to_datetime(forecast_date) + pd.Timedelta("1d")

    # cumulative deaths/cases 
    obs = data[place]['data'][start:][obs_field]
    end = obs.index.max()

    # predicted deaths/cases
    z = model.get(forecast_samples, forecast_field, forecast=True)
    
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
                   prefix="results",
                   target="deaths"):

    
    if places is None:
        places = list(data.keys())
    # Assemble performance metrics each place and time horizon
    details = pd.DataFrame()
    
    print(f'Scoring all places for {forecast_date} forecast')
    
    for place in tqdm(places):
                

        try:
            place_df = score_place(forecast_date,
                                   data,
                                   place,
                                   model_type=model_type,
                                   prefix=prefix,
                                   target=target)
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
        
        if len(places) > 1:
            summary.loc[date, 'horizon'] = horizon

            # Compute signed error / bias
            summary.loc[date, 'signed_err'] = rows['err'].mean()
        
            # Compute MAE
            summary.loc[date, 'MAE'] = rows['err'].abs().mean()
        
            # Compute avg. log-score
            summary.loc[date, 'log_score'] = rows['log_score'].mean()
        
            # Compute KS statistic
            ks, pval = scipy.stats.kstest(rows['quantile'], 'uniform')
            summary.loc[date,'KS'] = ks
            summary.loc[date,'KS_pval'] = pval

        else:
            summary.loc[date, 'horizon'] = horizon

            # Compute signed error / bias
            summary.loc[date, 'signed_err'] = rows['err']

            # Compute MAE
            summary.loc[date, 'MAE'] = rows['err']

            # Compute avg. log-score
            summary.loc[date, 'log_score'] = rows['log_score']
        
    summary['forecast_date'] = forecast_date
    
    return summary, details
