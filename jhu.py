import pandas as pd
from states import states, state_abb
import functools

@functools.lru_cache(128)
def load_and_massage(url):
    df = pd.read_csv(url)
    df = df.drop(columns=['Lat', 'Long'])
    df = df.rename(columns={'Province/State' : 'province', 'Country/Region' : 'country'})    
    df.province = df.province.replace(state_abb)
    df = df.set_index(['country', 'province'])
    df = df.T
    df.index = pd.to_datetime(df.index)
    return df

@functools.lru_cache(128)
def load_world():
    # Data files
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

    return df
        
@functools.lru_cache(128)
def load_us():
    
    df = load_world()
    
    # Get US data: index is now (province, variable)
    US = df.US

    
    # Get US data: index is now (province, variable)
    US = df.US

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