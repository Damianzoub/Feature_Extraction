from features.speed import compute_speed_from_position
from features.acceleration import acceleration_per_id
from features.rot import compute_rot
from features.trajectory import trajectory
from features.distance_and_straightness import _compute_total_and_straightness_metrics
from features.max_spatial_spread import compute_max_spatial_spread
from features.curvature import curvature_results
from utils.cache_utils import compute_or_load_feature
from features.zigzag import zigzag_index
from utils.shiptype_map import map_shiptype
from utils.clean_utils import clean_features
import os 

"""
self.col_kwargs = {
            "dataset_path":dataset_path,
            "time_col": time_col,
            "id_col": id_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "shiptype_col": shiptype_col
        }
"""
class FeaturePipeline:
    def __init__(self,**col_kwargs):
        self.col_kwargs = col_kwargs
        self.id_col = col_kwargs['id_col']
        self.time_col = col_kwargs['time_col']
        self.lat_col = col_kwargs['lat_col']
        self.lon_col = col_kwargs['lon_col']
        self.shiptype_col = col_kwargs['shiptype_col']
        self.dataset_name = os.path.splitext(os.path.basename(col_kwargs['dataset_path']))[0]
        


    def statistical_measures(self,data):
         speed = compute_or_load_feature("speed",self.dataset_name, lambda: compute_speed_from_position(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         acceleration = compute_or_load_feature( "acceleration",self.dataset_name,lambda: acceleration_per_id(data,self.time_col,self.id_col,speed_col="speed"))
         rot = compute_or_load_feature("rot",self.dataset_name,lambda: compute_rot(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         curvature = compute_or_load_feature('curvature',self.dataset_name,lambda: curvature_results(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         #zigzag = compute_or_load_feature("zigzag",self.dataset_name,lambda: zigzag_index(data,self.id_col,self.heading_col))
         #curvature['zigzag_index'] = curvature['mean_curvature']*zigzag['std_heading']
         ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
         ship_meta['shiptype'] = map_shiptype(ship_meta[self.shiptype_col])
         extract_df = (
              speed.merge(acceleration,on=self.id_col)
              .merge(rot,on=self.id_col)
              .merge(curvature,on=self.id_col)
              .merge(ship_meta[[self.id_col,'shiptype']],on=self.id_col,how='left')
         )
         return clean_features(extract_df)
    #Returns DataFrame with features per se
    def features_per_se(self,data):
         traj = compute_or_load_feature('traj',self.dataset_name,lambda: trajectory(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         distance_metrics = compute_or_load_feature('distance_metrics',self.dataset_name,lambda: _compute_total_and_straightness_metrics(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         max_spatial_spread = compute_or_load_feature( 'max_spatial_spread',self.dataset_name,lambda: compute_max_spatial_spread(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         #stop = compute_or_load_feature("stop",self.dataset_name, lambda: count_stops(data,self.id_col,self.time_col,self.lat_col,self.lon_col,self.speed_col))
         ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
         ship_meta['shiptype'] = map_shiptype(ship_meta[self.shiptype_col])
         extract_df= (
              traj.merge(distance_metrics,on=self.id_col)
              .merge(max_spatial_spread,on=self.id_col)
              #.merge(stop,on=self.id_col)
              .merge(ship_meta[[self.id_col,'shiptype']],on=self.id_col,how='left')
         )
         return clean_features(extract_df)
    
    def extract_all(self,data):
        speed = compute_or_load_feature("speed",self.dataset_name, lambda: compute_speed_from_position(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        acceleration = compute_or_load_feature( "acceleration",self.dataset_name,lambda: acceleration_per_id(data,self.time_col,self.id_col,speed_col="speed"))
        rot = compute_or_load_feature("rot",self.dataset_name,lambda: compute_rot(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        curvature = compute_or_load_feature('curvature',self.dataset_name,lambda: curvature_results(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        #zigzag = compute_or_load_feature("zigzag",self.dataset_name,lambda: zigzag_index(data,self.id_col,self.heading_col))
        traj = compute_or_load_feature('traj',self.dataset_name,lambda: trajectory(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        distance_metrics = compute_or_load_feature('distance_metrics',self.dataset_name,lambda: _compute_total_and_straightness_metrics(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        max_spatial_spread = compute_or_load_feature( 'max_spatial_spread',self.dataset_name,lambda: compute_max_spatial_spread(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        #stop = compute_or_load_feature("stop",self.dataset_name, lambda: count_stops(data,self.id_col,self.time_col,self.lat_col,self.lon_col,self.speed_col))
        #curvature['zigzag_index'] = curvature['mean_curvature']*zigzag['std_heading']
        ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
        ship_meta['shiptype'] = map_shiptype(ship_meta[self.shiptype_col])
        extract_df= (speed.merge(acceleration,on=self.id_col)
                .merge(rot,on=self.id_col)
                .merge(traj,on=self.id_col)
                .merge(distance_metrics,on=self.id_col)
                .merge(max_spatial_spread,on=self.id_col)
                .merge(curvature,on=self.id_col)
                #.merge(stop,on=self.id_col)
                .merge(ship_meta[[self.id_col,'shiptype']],on=self.id_col,how='left')
                )
        
        return clean_features(extract_df)
