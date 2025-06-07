import pandas as pd 
import datetime as dt

def categorize_time(df,time_col):
    if time_col not in df.columns:
        raise ValueError(f'{time_col} not in the dataset')
    df[time_col] = pd.to_datetime(df[time_col])
    df.loc[(df[time_col].dt.minute==59) & (df[time_col].dt.second==59),time_col] = df[time_col] + pd.Timedelta(seconds=1)
    return df
