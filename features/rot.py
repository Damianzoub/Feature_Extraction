import numpy as np
import pandas as pd
from utils.time_utils import categorize_time

def compute_rot(df, id_col, time_col, lat_col, lon_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[id_col, time_col])

    # Shifted lat/lon/time for computing heading
    df['lat_prev'] = df.groupby(id_col)[lat_col].shift(1)
    df['lon_prev'] = df.groupby(id_col)[lon_col].shift(1)
    df['lat_next'] = df.groupby(id_col)[lat_col].shift(-1)
    df['lon_next'] = df.groupby(id_col)[lon_col].shift(-1)
    df['time_prev'] = df.groupby(id_col)[time_col].shift(1)
    df['time_next'] = df.groupby(id_col)[time_col].shift(-1)

    # Calculate velocity components before (i) and after (i+1)
    time_diff_prev = (df[time_col] - df['time_prev']).dt.total_seconds().fillna(1)
    time_diff_prev = time_diff_prev.apply(lambda x: x + 0.000001 if x == 0 else x)
    
    time_diff_next = (df['time_next'] - df[time_col]).dt.total_seconds().fillna(1)
    time_diff_next = time_diff_next.apply(lambda x: x + 0.000001 if x == 0 else x)

    df['vx_i'] = (df[lon_col] - df['lon_prev']) / time_diff_prev
    df['vy_i'] = (df[lat_col] - df['lat_prev']) / time_diff_prev
    df['vx_ip1'] = (df['lon_next'] - df[lon_col]) / time_diff_next
    df['vy_ip1'] = (df['lat_next'] - df[lat_col]) / time_diff_next

    # Compute heading angles in radians
    heading_i = np.arctan2(df['vy_i'], df['vx_i'])
    heading_ip1 = np.arctan2(df['vy_ip1'], df['vx_ip1'])

    # Compute change in heading (ensure continuity with np.unwrap)
    df['heading_diff'] = np.unwrap(heading_ip1 - heading_i)

    # Time difference for ROT (again, avoid divide-by-zero)
    df['rot_time_diff'] = time_diff_next

    # Compute ROT (radians per second)
    df['ROT'] = df['heading_diff'] / df['rot_time_diff']

    # Aggregate stats
    return df.groupby(id_col)['ROT'].agg(
        avg_rot='mean',
        std_rot='std'
    ).reset_index()
