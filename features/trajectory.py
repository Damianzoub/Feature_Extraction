import pandas as pd 


def trajectory(df,id_col,time_col,lat_col,long_col):
    new_df = df.copy()
    new_df[time_col] = pd.to_datetime(new_df[time_col])
    grouped = new_df.sort_values([id_col,time_col]).groupby(id_col)

    features = grouped.agg(
        start_lat=(lat_col, 'first'),
        start_long=(long_col, 'first'),
        end_lat=(lat_col, 'last'),
        end_long=(long_col, 'last'),
        start_time=(time_col, 'first'),
        end_time=(time_col, 'last')
    )
    #Travelling Time and Location Information for every feature
    features['duration_second'] = (features['end_time']-features['start_time']).dt.total_seconds()
    features['duration_hour'] = features['duration_second']/3600
    features['start_year'] = features['start_time'].dt.year
    features['start_month'] = features['start_time'].dt.month
    features['start_day'] = features['start_time'].dt.day
    features['start_hour'] = features['start_time'].dt.hour
    features['start_minute'] = features['start_time'].dt.minute
    features['end_year'] = features['end_time'].dt.year
    features['end_month'] = features['end_time'].dt.month
    features['end_day'] = features['end_time'].dt.day
    features['end_hour'] = features['end_time'].dt.hour
    features['end_minute'] = features['end_time'].dt.minute
    
    features = features.reset_index()
    return features[[self.id_col, 'start_lat', 'start_lon', 'end_lat', 'end_lon',
           'start_time', 'end_time', 'duration_second', 'duration_hour',
           'start_year', 'start_month', 'start_day', 'start_hour', 'start_minute',
           'end_year', 'end_month', 'end_day', 'end_hour', 'end_minute'
           ]]
