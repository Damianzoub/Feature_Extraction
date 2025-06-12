import pandas as pd

def count_stops(df,id_col,time_col,lat_col,lon_col,speed_col,stop_speed_threshold=0.5,min_stop_duration=300):
    """
        Group data by shipid and compute the stops for every shipid
    """
    required_cols = [id_col,time_col,lat_col,lon_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing one or more required columns")
    new_df = df.sort_values(by=[id_col,time_col])

    result = new_df.groupby(id_col).apply(
        lambda group: _compute_stops(group,time_col,speed_col,stop_speed_threshold,min_stop_duration)
    ).reset_index()

    return result[[id_col,'num_stops']]


def _compute_stops(group,time_col,speed_col,stop_speed_threshold,min_stop_duration):
    stop_count = 0
    in_stop = False
    stop_start_time = None 

    for i, row in group.iterrows():
        if row[speed_col] <= stop_speed_threshold:
            if not in_stop:
                in_stop= True 
                stop_start_time = row[time_col]
            else:
                if in_stop:
                    stop_duration = (row[time_col]-stop_start_time).total_seconds()
                    if stop_duration >= min_stop_duration:
                        stop_duration +=1
                    in_stop = False
        
    if in_stop:
        stop_duration = (group.iloc[-1][time_col]-stop_start_time).total_seconds()
        if stop_duration >= min_stop_duration:
            stop_count+=1
        
    return pd.Series({'num_stops':stop_count})
