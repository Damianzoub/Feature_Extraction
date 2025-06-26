from utils.time_utils import categorize_time
import pandas as pd
def acceleration_per_id(df, time_col, id_col, speed_col='speed'):
    df= categorize_time(df,time_col)
    df = df.copy()
    df = df.sort_values(by=[id_col, time_col])

    
    df['time_diff'] = df.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(1) 
    df['time_diff'] = df['time_diff'].apply(lambda x: x+0.000001 if x==0 else x)
    df['speed_diff'] = df.groupby(id_col)[speed_col].diff().fillna(0)

    df['acceleration'] = df['speed_diff'] / df['time_diff']

    return df.groupby(id_col)['acceleration'].agg(
        avg_acceleration='mean',
        max_acceleration='max',
        min_acceleration='min',
        std_acceleration='std'
    ).reset_index()
