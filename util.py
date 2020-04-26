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

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Diamond Princess':'Diamond Princess',
    'Grand Princess':'Grand Princess'
}


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


def load_state_data(source="covidtracker"):

    # US state data
    if source=="covidtracker":
        US = covidtracking.load_us()
    if source=="jhu":
        US = get_jhu_US_data_new_format()
    
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


def get_jhu_US_data_new_format():
    baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    def loadData(fileName, columnName):
        data = pd.read_csv(baseURL + fileName)
        return (data)
    confirmed = loadData(
    "time_series_covid19_confirmed_US.csv", "confirmed")
    confirmed = confirmed.drop(columns=['UID','Lat', 'Long_',
                                "iso2","iso3","code3","FIPS",
                                "Admin2", "Country_Region","Combined_Key"])
    confirmed = confirmed.groupby('Province_State').sum().T
    confirmed = confirmed.rename(columns=us_state_abbrev)  
    confirmed =confirmed.reset_index()
    confirmed = confirmed.rename(columns={'index': 'date'})


    confirmed['date'] = pd.to_datetime(confirmed['date'], infer_datetime_format=False) 
    deaths = loadData(
    "time_series_covid19_deaths_US.csv", "deaths")
    deaths = deaths.drop(columns=['UID','Lat', 'Long_',
                                "iso2","iso3","code3","FIPS",
                                "Admin2", "Country_Region","Combined_Key","Population"])
    deaths = deaths.groupby('Province_State').sum().T
    deaths = deaths.rename(columns=us_state_abbrev)
    
    deaths= deaths.reset_index()
    deaths = deaths.rename(columns={'index': 'date'})
    df = pd.concat([deaths,confirmed],axis=1,keys=('death','confirmed'))

    df = df.reorder_levels([1,0], axis=1).sort_index(axis=1)
   
    df = df.set_index(confirmed['date'])
   
    return df

    
    
    

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
              confirmed_min = 10,
              death_min = 5,
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
                  show = True, 
                  **kwargs):
    
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
            
    fig = models.plot_R0(mcmc_samples, start_)    
    plt.title(place)
    plt.tight_layout()
    
    if save:
        filename = f'{save_path}/{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()