import pandas as pd
import cachetools.func

@cachetools.func.ttl_cache(ttl=3600)
def uga_traits():
    url = 'https://raw.githubusercontent.com/CEIDatUGA/COVID-19-DATA/master/US/US_state_traits.csv'
    traits = pd.read_csv(url, thousands = ',')
    traits = traits.set_index('postalCode')
    traits.index.name = 'state'
    traits.index = traits.index.fillna('tot')  # there is a total line which has values for a few columns
    not_empty = [not traits[c].isnull().all() for c in traits.columns]
    traits = traits.loc[:, not_empty]
    traits['popdensity'] = traits.totalpop / traits.Land_Area_mile2
    
    # repeat state as columns for convenience
    traits.insert(0, 'state', traits.index)

    return traits
    
    
@cachetools.func.ttl_cache(ttl=3600)
def uga_interventions():
    url = 'https://github.com/CEIDatUGA/COVID-19-DATA/raw/master/US/us-state-intervention-data/US_state_intervention_time_series.csv'
    df = pd.read_csv(url, index_col=0)
    df.NAME = df.NAME.replace(abbrev)
    df = df.rename(columns={'NAME' : 'state', 'DATE' : 'date'})
    df.date = pd.to_datetime(df.date)
    df = df.set_index(['state', 'date'])
    
    # repeat state and date as columns (vs. index) for convenience
    df.insert(0, 'state', df.index.get_level_values(0))
    df.insert(1, 'date', df.index.get_level_values(1))
    return df

def local_traits():
    '''Read state traits from a downloaded data file. Currently using uga_traits instead'''
    traits = pd.read_csv('state-pop.csv')
    traits['id'] = [states.abbrev[state] for state in traits.State]
    traits = traits.set_index('id')
    return traits

states = { # includes DC
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

territories = {
        'AS': 'American Samoa',
        'GU': 'Guam',
        'MP': 'Northern Mariana Islands',
        'NA': 'National',
        'PR': 'Puerto Rico',
        'VI': 'Virgin Islands'
}

states_territories = dict(states, **territories)

abbrev = {v: k for k, v in states_territories.items()}