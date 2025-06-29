import pandas as pd 
import geopandas as gpd
from movingpandas import TrajectoryCollection

def count_stops(df,id_col,time_col,lat_col,lon_col,min_stop_duration='5min'):
        """
        Returns:
            - DataFrame with columns [id_col,'stop_count']
        """
        
        new_df = df.copy()
        new_df[time_col] = pd.to_datetime(new_df[time_col])
        
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col],df[lat_col]),
            crs='EPSG:4326'
        )
        gdf = gdf.sort_values(by=[id_col,time_col])
        
        traj_col = TrajectoryCollection(gdf,traj_id_col=id_col,t=time_col)
        min_duration = pd.Timedelta(min_stop_duration)
        stops_summary = []
        for traj in traj_col.trajectories:
            stops = traj.get_stops(min_duration=min_duration)
            stops_summary.append({
                id_col:traj.id,
                "stop_count":len(stops)
            })
        return pd.DataFrame(stops_summary)
