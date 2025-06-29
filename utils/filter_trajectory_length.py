import pandas as pd

def filter_trajectories_by_length(df, id_col='shipid', min_points=5):
    """
    Φιλτράρει το DataFrame κρατώντας μόνο τροχιές που έχουν τουλάχιστον min_points σημεία.

    Args:
        df (pd.DataFrame): Το DataFrame με τις τροχιές.
        id_col (str): Η στήλη που περιέχει τα ID των τροχιών (π.χ. 'shipid').
        min_points (int): Ο ελάχιστος αριθμός σημείων που απαιτούνται για κάθε τροχιά.

    Returns:
        pd.DataFrame: Φιλτραρισμένο DataFrame.
    """
    if id_col not in df.columns:
        raise ValueError(f"Η στήλη '{id_col}' δεν υπάρχει στο DataFrame.")

    counts = df.groupby(id_col).size()
    valid_ids = counts[counts >= min_points].index
    filtered_df = df[df[id_col].isin(valid_ids)].copy()
    return filtered_df
