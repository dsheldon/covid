import pandas as pd
import cachetools.func
import warnings

from . import states

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
    'death' : 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
}

    # Load each data file into a dataframe with row index = date, and column index = (country, province)
    d = {key: load_and_massage(url) for key, url in sources.items()}

    # Concatenate data frames: column index is now (variable, country, province)
    df = pd.concat(d.values(), axis=1, keys=d.keys())

    # Permute order of index to (country, province, variable) and sort the columns by the index value
    df = df.reorder_levels([1,2,0], axis=1).sort_index(axis=1)

    return df
        
#@functools.lru_cache(128)
@cachetools.func.ttl_cache(ttl=3600)
def load_us():
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
    confirmed = confirmed.rename(columns=states.abbrev)  
    confirmed =confirmed.reset_index()
    confirmed = confirmed.rename(columns={'index': 'date'})


    confirmed['date'] = pd.to_datetime(confirmed['date'], infer_datetime_format=False) 
    deaths = loadData(
    "time_series_covid19_deaths_US.csv", "death")
    deaths = deaths.drop(columns=['UID','Lat', 'Long_',
                                "iso2","iso3","code3","FIPS",
                                "Admin2", "Country_Region","Combined_Key","Population"])
    deaths = deaths.groupby('Province_State').sum().T
    deaths = deaths.rename(columns=states.abbrev)
    
    deaths= deaths.reset_index()
    deaths = deaths.rename(columns={'index': 'date'})
    df = pd.concat([deaths,confirmed],axis=1,keys=('death','confirmed'))

    df = df.reorder_levels([1,0], axis=1).sort_index(axis=1)
   
    df = df.set_index(confirmed['date'])
   
    return df