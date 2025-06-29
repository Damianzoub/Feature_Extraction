import numpy as np
import pandas as pd

def clean_features(df: pd.DataFrame,fill_value: float = 0.0) -> pd.DataFrame:

    df.replace([np.inf,-np.inf,np.nan],fill_value,inplace=True)
    return df
