import pandas as pd
import states
#import functools
import cachetools.func
import warnings

#@functools.lru_cache(128)
@cachetools.func.ttl_cache(ttl=3600)
def load_and_massage(url):
    df = pd.read_csv(url)
    df = df.drop(columns=['Lat', 'Long'])
    df = df.rename(columns={'Province/State' : 'province', 'Country/Region' : 'country'})    
    df.province = df.province.replace(states.abbrev)
    df.province = df.province.fillna('tot')
    df = df.set_index(['country', 'province'])
    df = df.T
    df.index = pd.to_datetime(df.index)
    return df

#@functools.lru_cache(128)
@cachetools.func.ttl_cache(ttl=3600)
def load_world():

    sources = {
    'confirmed' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    'deaths' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
}

    # Load each data file into a dataframe with row index = date, and column index = (country, province)
    d = {key: load_and_massage(url) for key, url in sources.items()}

    # Concatenate data frames: column index is now (variable, country, province)
    df = pd.concat(d.values(), axis=1, keys=d.keys())

    # Permute order of index to (country, province, variable) and sort the columns by the index value
    df = df.reorder_levels([1,2,0], axis=1).sort_index(axis=1)

    return df
    
    return df
        
#@functools.lru_cache(128)
@cachetools.func.ttl_cache(ttl=3600)
def load_us():

    warnings.warn('This data is now deprecated. Use jhu.load_world() for country-level data'
            'or covidtracking.load_us() for state-level data in the US')
    
    sources = {
        'confirmed' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
        'deaths' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
        'recovered' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
    }

    # Load each data file into a dataframe with row index = date, and column index = (country, province)
    d = {key: load_and_massage(url) for key, url in sources.items()}

    # Concatenate data frames: column index is now (variable, country, province)
    df = pd.concat(d.values(), axis=1, keys=d.keys())

    # Permute order of index to (country, province, variable) and sort the columns by the index value
    df = df.reorder_levels([1,2,0], axis=1).sort_index(axis=1)
    
    # Get US data: index is now (province, variable)
    US = df.US

    
    # Get US data: index is now (province, variable)
    US = df.US

    # Maps place name in the US to states. Place names could be state
    # abbreviations ('MA') or city/state combinations ('Boston, MA')
    #
    
    # Input is a key like ('Boston, MA', 'confirmed') or ('MA', 'confirmed')
    #  Output in both cases should be ('MA', 'confirmed')
    def by_state(k):
        s,v=k
        return s.split(',')[-1].strip(), v

    US = US.groupby(by_state, axis=1, as_index=True).sum()
    US.columns = pd.MultiIndex.from_tuples(US.columns)
    
    # Filter for province in states.keys()
    is_state = [province in states.keys() for province in US.columns.get_level_values(0) ]
    US = US.iloc[:, is_state]
    
    # Add total column
    US_tot = US.groupby(level=1, axis=1).sum()
    for col in US_tot.columns:
        US['tot', col] = US_tot[col]
    
    return US