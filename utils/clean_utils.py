import numpy as np
import pandas as pd

def clean_features(df: pd.DataFrame,fill_value: float = 0.0) -> pd.DataFrame:

    numeric_df = df.select_dtypes(include=[np.number])
    has_nan = df.isnull().values.any()
    has_inf = np.isinf(numeric_df.values).any()
    if not has_nan or not has_inf:
        return df

    df.replace([-np.inf,np.inf],np.nan,inplace=True)
    df.fillna(fill_value,inplace=True)
    return df
