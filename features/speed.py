import pandas as pd
from utils.time_utils import categorize_time

def average_speed_per_id(df,id_col,time_col,speed_col):
    df = categorize_time(df,time_col)
    new_df = df.copy()

    aggregate = new_df.groupby([id_col,time_col])[speed_col].agg(
        avg_speed='mean',
        max_speed='max',
        min_speed='min',
        std_speed='std'
    ).reset_index()

    return aggregate.groupby(id_col).agg(
        {
            "avg_speed":"mean",
            "max_speed":"max",
            "min_speed":"min",
            "std_speed":"std"
        }
    ).reset_index()