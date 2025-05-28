import pandas as pd 
from utils.time_utils import categorize_time


def acceleration_per_id(df,time_col,id_col,speed_col):
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not in dataset")
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not in dataset")
    if speed_col not in df.columns:
        raise ValueError(f"{speed_col} not in dataset")
    df = categorize_time(df,time_col)
    new_df = df.copy()
    new_df = new_df.sort_values(by=[id_col,time_col])

    new_df['time_diff'] = new_df.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(1)
    new_df['speed_diff'] = new_df.groupby(id_col)[speed_col].diff().fillna(0)
    new_df['acceleration'] = new_df['speed_diff']/new_df['time_diff']

    return new_df.groupby(id_col)['acceleration'].agg(
        avg_acceleration = 'mean',
        max_acceleration='max',
        min_acceleration='min',
        std_acceleration='std'
    ).reset_index()