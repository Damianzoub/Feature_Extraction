import numpy as np
import pandas as pd
from geopy.distance import geodesic
from utils.time_utils import categorize_time

def compute_speed_from_position(df, id_col, time_col, lat_col, lon_col):
    df = categorize_time(df, time_col)
    df = df.copy()
    df = df.sort_values(by=[id_col, time_col])

    # Shift coordinates and time
    df['lat_prev'] = df.groupby(id_col)[lat_col].shift(1)
    df['lon_prev'] = df.groupby(id_col)[lon_col].shift(1)
    df['time_prev'] = df.groupby(id_col)[time_col].shift(1)

    # Compute distance in meters using geodesic
    df['dist_m'] = df.apply(lambda row: geodesic((row['lat_prev'], row['lon_prev']), (row[lat_col], row[lon_col])).meters
                            if pd.notnull(row['lat_prev']) else 0, axis=1)

    # Compute time difference in seconds
    df['time_diff'] = (df[time_col] - df['time_prev']).dt.total_seconds().fillna(1)
    df['time_diff'] = df['time_diff'].apply(lambda x: x+0.000001 if x==0 else x)
    # Speed = distance / time (in m/s)
    df['speed'] = df['dist_m'] / df['time_diff']

    return df.groupby(id_col)['speed'].agg(
        avg_speed='mean',
        max_speed='max',
        min_speed='min',
        std_speed='std'
    ).reset_index()


