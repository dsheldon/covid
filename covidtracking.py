import numpy as np
import pandas as pd

def load_us():
    df = pd.read_csv('https://covidtracking.com/api/states/daily.csv')
    df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df = df.set_index('date')
    df = df.drop(columns=['dateChecked'])
    df = df.pivot(columns='state')
    df.columns = df.columns.swaplevel()
    df.sort_index(axis=1, inplace=True)
    return df