import pandas as pd
import cachetools.func

def get_url(sheet):
    book = 'https://docs.google.com/spreadsheets/d/1_mk1J3ElMH5EmPILcrx8s1KqHqdQYXCWroJeKTR3pV4'
    return f'{book}/gviz/tq?tqx=out:csv&sheet={sheet}'

@cachetools.func.ttl_cache(ttl=3600)
def uga_traits():
    traits = pd.read_csv(get_url('state_traits'), thousands = ',')
    traits = traits.set_index('postalCode')
    traits.index = traits.index.fillna('tot')  # there is a total line which has values for a few columns
    not_empty = [not traits[c].isnull().all() for c in traits.columns]
    traits = traits.loc[:, not_empty]
    traits['popdensity'] = traits.totalpop / traits.Land_Area_mile2
    return traits
    
@cachetools.func.ttl_cache(ttl=3600)
def uga_announcements():
    announcements = pd.read_csv(get_url('state-covid-announcements'))
    return announcements


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


state_abb = {v: k for k, v in states.items()}