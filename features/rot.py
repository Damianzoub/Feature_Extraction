import numpy as np
from utils.time_utils import categorize_time
import pandas as pd 
def compute_rot(df, id_col, time_col, lat_col, lon_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[id_col, time_col])

    # Compute velocities for heading
    df['lat_prev'] = df.groupby(id_col)[lat_col].shift(1)
    df['lon_prev'] = df.groupby(id_col)[lon_col].shift(1)
    df['lat_next'] = df.groupby(id_col)[lat_col].shift(-1)
    df['lon_next'] = df.groupby(id_col)[lon_col].shift(-1)
    df['time_prev'] = df.groupby(id_col)[time_col].shift(1)
    df['time_next'] = df.groupby(id_col)[time_col].shift(-1)

    # Heading i
    df['vx_i'] = (df[lon_col] - df['lon_prev']) / ((df[time_col] - df['time_prev']).dt.total_seconds()+0.01)
    df['vy_i'] = (df[lat_col] - df['lat_prev']) / ((df[time_col] - df['time_prev']).dt.total_seconds()+ 0.01)
    heading_i = np.arctan2(df['vy_i'], df['vx_i'])

    # Heading i+1
    df['vx_ip1'] = (df['lon_next'] - df[lon_col]) / ((df['time_next'] - df[time_col]).dt.total_seconds()+0.01)
    df['vy_ip1'] = (df['lat_next'] - df[lat_col]) / ((df['time_next'] - df[time_col]).dt.total_seconds()+0.01)
    heading_ip1 = np.arctan2(df['vy_ip1'], df['vx_ip1'])

    # Rate of Turn (ROT)
    df['heading_diff'] = heading_ip1 - heading_i
    df['rot_time_diff'] = (df['time_next'] - df[time_col]).dt.total_seconds()+0.01
    df['ROT'] = df['heading_diff'] / df['rot_time_diff']

    return df.groupby(id_col)['ROT'].agg(
        avg_rot='mean',
        std_rot='std'
    ).reset_index()
