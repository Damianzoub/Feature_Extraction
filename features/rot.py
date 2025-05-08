from utils.time_utils import categorize_time

def rot_per_id(df,head_col,id_col,time_col):
    if head_col not in df.columns:
        raise ValueError(f'{head_col} not in dataset')
    if time_col not in df.columns:
        raise ValueError(f'{time_col} not in dataset')
    if id_col not in df.columns:
        raise ValueError(f'{id_col} not in dataset')
    
    df = categorize_time(df,time_col)
    new_df = df.copy()
    new_df = new_df.sort_values(by=[id_col,time_col])
    new_df['heading_diff'] = new_df.groupby(id_col)[head_col].diff()
    new_df['time_diff'] = new_df.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(1)
    new_df['ROT'] =new_df['heading_diff']/new_df['time_diff']
    return new_df.groupby(id_col)['ROT'].agg(
        rot_mean='mean',
        rot_std = 'std'
    ).reset_index()
