import pandas as pd
import numpy as np
from utils.time_utils import categorize_time

def compute_speed_from_position(df, id_col, time_col, lat_col, lon_col):
    df = categorize_time(df,time_col)
    new_df = df.copy()
    new_df = new_df.sort_values(by=[id_col, time_col])

    # Shift lat, lon, and time to get the previous point
    new_df['lat_prev'] = new_df.groupby(id_col)[lat_col].shift(1)
    new_df['lon_prev'] = new_df.groupby(id_col)[lon_col].shift(1)
    new_df['time_prev'] = new_df.groupby(id_col)[time_col].shift(1)

    # Compute Euclidean distance in degrees (optionally convert to meters using haversine or geopy)
    new_df['dist'] = np.sqrt((new_df[lat_col] - new_df['lat_prev'])**2 + 
                             (new_df[lon_col] - new_df['lon_prev'])**2)

    # Compute time difference in seconds
    new_df['time_diff'] = (new_df[time_col] - new_df['time_prev']).dt.total_seconds().fillna(1) 
    new_df['time_diff'] = new_df['time_diff'].apply(lambda x: x+0.000001 if x==0 else x)

    # Compute speed
    new_df['speed'] = new_df['dist'] / new_df['time_diff']

    # Group by ID to compute aggregate statistics
    return new_df.groupby(id_col)['speed'].agg(
        avg_speed='mean',
        max_speed='max',
        min_speed='min',
        std_speed='std'
    ).reset_index()
