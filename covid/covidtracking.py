import numpy as np
import pandas as pd
import itertools
import cachetools.func

@cachetools.func.ttl_cache(ttl=3600)
def load_us():
    df = pd.read_csv('https://covidtracking.com/api/states/daily.csv')
    df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df = df.set_index('date')
    df = df.drop(columns=['dateChecked'])
    df['confirmed'] = df['positive'] # useful alias to match other data
    df = df.pivot(columns='state')
    df.columns = df.columns.swaplevel()
    df.sort_index(axis=1, inplace=True)
    return df


@cachetools.func.ttl_cache(ttl=3600)
def load_us_flat(start='2020-03-04'):
    
    df = pd.read_csv('https://covidtracking.com/api/states/daily.csv')
    df.date = pd.to_datetime(df.date, format='%Y%m%d')
    
    df = df.set_index(['state', 'date'])
    df.sort_index(axis=0, inplace=True)
    
    states = df.index.unique(level=0)
    days = df.index.unique(level=1)
    full_index = itertools.product(states, days)
    df = df.reindex(full_index)
    df.sort_index(axis=0, inplace=True)

    end = None
    df = df.loc[(slice(None),slice(start, end)),:]

    return df
