import numpy as np
import pandas as pd
#if the features that we extract have some np.nan values or np.inf
def clean_features(df: pd.DataFrame,fill_value: float = 0.0) -> pd.DataFrame:
    df.replace([-np.inf,np.inf],np.nan,inplace=True)
    df.fillna(fill_value,inplace=True)
    return df
  
