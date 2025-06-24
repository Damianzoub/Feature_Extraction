import pandas as pd

def zigzag_index(df,id_col,heading_col):
    important_cols = [id_col,heading_col]
    if not all(col in df.columns for col in important_cols):
        raise ValueError("Not column found")
    
    heading_std = df.groupby(id_col)[heading_col].std().reset_index()
    heading_std.rename(columns={heading_col:'std_heading'},inplace=True)

    return heading_std
