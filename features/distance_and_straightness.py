from geopy.distance import geodesic
import pandas as pd

def _compute_total_and_straightness_metrics(df,id_col,time_col,lat_col,lon_col):
        required_cols = [id_col, time_col, lat_col, lon_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing one or more required columns in the dataset.")

        new_df = df.sort_values(by=[id_col, time_col])

        result = new_df.groupby(id_col).apply(
            lambda group: _compute_metrics(group,lat_col=lat_col,lon_col=lon_col)
        ).reset_index()

        return result[[id_col, 'total_distance_km', 'straightness_ratio', 'tortuosity']]

def _compute_metrics(group,lat_col,lon_col):
        coords = list(zip(group[lat_col], group[lon_col]))
        total_distance = sum(
            geodesic(coords[i], coords[i + 1]).kilometers
            for i in range(len(coords) - 1)
        ) if len(coords) >= 2 else 0.0

        if len(coords) < 2:
            straightness, tortuosity = 0.0, 0.0
        else:
            direct_distance = geodesic(coords[0], coords[-1]).kilometers
            straightness = 0 if total_distance == 0 else direct_distance / total_distance
            tortuosity = 0 if direct_distance == 0 else total_distance / direct_distance

        return pd.Series({
            'total_distance_km': total_distance,
            'straightness_ratio': straightness,
            'tortuosity': tortuosity
        })
