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
    features['duration_second'] = (features['end_time']-features['start_time']).dt.total_seconds()
    features['duration_hour'] = features['duration_second']/3600
    features = features.reset_index()
    return features[[id_col,'start_lat','start_long','end_lat',
                          'end_long','start_time','end_time','duration_second'
                          ,'duration_hour']]
