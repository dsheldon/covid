import pandas as pd
import cachetools.func
import warnings

from . import states

@cachetools.func.ttl_cache(ttl=600)
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

@cachetools.func.ttl_cache(ttl=600)
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
        
@cachetools.func.ttl_cache(ttl=600)
def get_fips_codes():
    '''Get valid FIPS codes from covid19forecasthub'''
    url = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv'
    df = pd.read_csv(url)

    fips_codes = df['location']
    fips_codes = fips_codes.loc[fips_codes != 'US'].astype(int)
    return fips_codes

def filter_counties(df):
    '''Filter rows from JHU data schema to counties represented in forecast hub'''
    fips_codes = get_fips_codes()
    
#     exclude_counties = ['Kings, New York, US', 
#                         'Queens, New York, US', 
#                         'Bronx, New York, US', 
#                         'Richmond, New York, US']

    exclude_counties = []
    
    # Subset to locations: 
    #   (1) in US,
    #   (2) with county name, 
    #   (3) with FIPS code recognized by forecast hub
    #   (4) not in list of NYC counties with no data on JHU
    
    df = df.loc[(df['iso2']=='US') & (df['Admin2']) & (df['FIPS'])].copy()
    df['FIPS'] = df['FIPS'].astype(int)
    df = df.loc[df['FIPS'].isin(fips_codes)].copy()
    df = df.loc[~df['Combined_Key'].isin(exclude_counties)].copy()
    
    return df
    
@cachetools.func.ttl_cache(ttl=600)
def get_county_info():
    '''Get state info from JHU location lookup file'''
    
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
    df = pd.read_csv(url)
    
    df = filter_counties(df)

    # Add county and state columns, and set key to <state abbrev>-<county name>
    df['name'] = df['Admin2']
    df['state'] = df['Province_State'].replace(states.abbrev)
    df['key'] = df['state'] + '-' + df['name']
    df = df.set_index('key')
    return df

@cachetools.func.ttl_cache(ttl=600)
def get_state_info():
    '''Get state info from JHU location lookup file'''
    
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
    df = pd.read_csv(url)
    df = df.loc[df['FIPS'] <= 78].copy()
    df['name'] = df['Province_State']
    df['key'] = df['Province_State'].replace(states.abbrev)
    df = df.set_index('key')
    return df


@cachetools.func.ttl_cache(ttl=600)
def load_us(counties=False):
    
    baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

    def load_us_time_series(file):
        '''Load data in JHU US time series format (death or confirmed)'''
    
        df = pd.read_csv(baseURL + file)

        meta_cols = ['UID',
                     'Lat',
                     'Long_',
                     'iso2',
                     'iso3',
                     'code3',
                     'FIPS',
                     'Admin2',
                     'Province_State',
                     'Country_Region',
                     'Combined_Key',
                     'Population']

        meta_cols = [c for c in meta_cols if c in df.columns]

        if counties:
            # subset to valid counties, set index to <state abbrev>-<county> and drop other metadata columns
            df = filter_counties(df)
            state = df['Province_State'].replace(states.abbrev)
            county = df['Admin2']
            #county = county.replace({'New York': 'New York City'}) # correct inconsistency with metadata table
            df = df.drop(columns=meta_cols)
            df = df.set_index(state + '-' + county)

        else:
            # group by state
            df['state'] = df['Province_State'].replace(states.abbrev)
            df = df.drop(columns=meta_cols).groupby('state').sum()

        df = df.T
        df.index = pd.to_datetime(df.index)
        
        return df

    
    confirmed = load_us_time_series("time_series_covid19_confirmed_US.csv")
    deaths = load_us_time_series("time_series_covid19_deaths_US.csv")
    
    # Combine deaths and confirmed
    df = pd.concat([deaths,confirmed],axis=1,keys=('death','confirmed'))
    df = df.reorder_levels([1,0], axis=1).sort_index(axis=1)
    
    return(df)