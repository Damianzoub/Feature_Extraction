from features.speed import average_speed_per_id
from features.acceleration import acceleration_per_id
from features.rot import rot_per_id
from features.trajectory import trajectory
from features.distance_and_straightness import _compute_total_and_straightness_metrics
from features.max_spatial_spread import compute_max_spatial_spread
from features.curvature import curvature_results
from features.stops import count_stops
from utils.cache_utils import compute_or_load_feature
from features.zigzag import zigzag_index
from utils.shiptype_map import map_shiptype
import os 

"""
self.col_kwargs = {
            "dataset_path":dataset_path,
            "time_col": time_col,
            "id_col": id_col,
            "speed_col": speed_col,
            "heading_col": heading_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "course_col": course_col,
            "shiptype_col": shiptype_col,
            "destination_col": destination_col
        }
"""
class FeaturePipeline:
    def __init__(self,**col_kwargs):
        self.col_kwargs = col_kwargs
        self.id_col = col_kwargs['id_col']
        self.time_col = col_kwargs['time_col']
        self.speed_col = col_kwargs['speed_col']
        self.heading_col = col_kwargs['heading_col']
        self.lat_col = col_kwargs['lat_col']
        self.lon_col = col_kwargs['lon_col']
        self.course_col = col_kwargs['course_col']
        self.shiptype_col = col_kwargs['shiptype_col']
        self.destination_col = col_kwargs['destination_col']
        self.dataset_name = os.path.splitext(os.path.basename(col_kwargs['dataset_path']))[0]
        


    def statistical_measures(self,data):
         speed = compute_or_load_feature("speed",self.dataset_name, lambda: average_speed_per_id(data,self.id_col,self.time_col,self.speed_col))
         acceleration = compute_or_load_feature( "acceleration",self.dataset_name,lambda: acceleration_per_id(data,self.time_col,self.id_col,self.speed_col))
         rot = compute_or_load_feature("rot",self.dataset_name,lambda: rot_per_id(data,self.heading_col,self.id_col,self.time_col))
         curvature = compute_or_load_feature('curvature',self.dataset_name,lambda: curvature_results(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         zigzag = compute_or_load_feature("zigzag",self.dataset_name,lambda: zigzag_index(data,self.id_col,self.heading_col))
         curvature['zigzag_index'] = curvature['mean_curvature']*zigzag['std_heading']
         ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
         ship_meta['shiptype_label'] = map_shiptype(ship_meta[self.shiptype_col])
         return (
              speed.merge(acceleration,on=self.id_col)
              .merge(rot,on=self.id_col)
              .merge(curvature,on=self.id_col)
              .merge(ship_meta[[self.id_col,'shiptype_label']],on=self.id_col,how='left')
         )
    #Returns DataFrame with features per se
    def features_per_se(self,data):
         traj = compute_or_load_feature('traj',self.dataset_name,lambda: trajectory(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         distance_metrics = compute_or_load_feature('distance_metrics',self.dataset_name,lambda: _compute_total_and_straightness_metrics(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         max_spatial_spread = compute_or_load_feature( 'max_spatial_spread',self.dataset_name,lambda: compute_max_spatial_spread(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
         #stop = compute_or_load_feature("stop",self.dataset_name, lambda: count_stops(data,self.id_col,self.time_col,self.lat_col,self.lon_col,self.speed_col))
         ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
         ship_meta['shiptype_label'] = map_shiptype(ship_meta[self.shiptype_col])
         return (
              traj.merge(distance_metrics,on=self.id_col)
              .merge(max_spatial_spread,on=self.id_col)
              #.merge(stop,on=self.id_col)
              .merge(ship_meta[[self.id_col,'shiptype_label']],on=self.id_col,how='left')
         )
    
    def extract_all(self,data):
        speed = compute_or_load_feature("speed",self.dataset_name, lambda: average_speed_per_id(data,self.id_col,self.time_col,self.speed_col))
        acceleration = compute_or_load_feature( "acceleration",self.dataset_name,lambda: acceleration_per_id(data,self.time_col,self.id_col,self.speed_col))
        rot = compute_or_load_feature("rot",self.dataset_name,lambda: rot_per_id(data,self.heading_col,self.id_col,self.time_col))
        curvature = compute_or_load_feature('curvature',self.dataset_name,lambda: curvature_results(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        zigzag = compute_or_load_feature("zigzag",self.dataset_name,lambda: zigzag_index(data,self.id_col,self.heading_col))
        traj = compute_or_load_feature('traj',self.dataset_name,lambda: trajectory(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        distance_metrics = compute_or_load_feature('distance_metrics',self.dataset_name,lambda: _compute_total_and_straightness_metrics(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        max_spatial_spread = compute_or_load_feature( 'max_spatial_spread',self.dataset_name,lambda: compute_max_spatial_spread(data,self.id_col,self.time_col,self.lat_col,self.lon_col))
        #stop = compute_or_load_feature("stop",self.dataset_name, lambda: count_stops(data,self.id_col,self.time_col,self.lat_col,self.lon_col,self.speed_col))
        curvature['zigzag_index'] = curvature['mean_curvature']*zigzag['std_heading']
        ship_meta = data[[self.id_col, self.shiptype_col]].drop_duplicates()
        ship_meta['shiptype_label'] = map_shiptype(ship_meta[self.shiptype_col])
        return (speed.merge(acceleration,on=self.id_col)
                .merge(rot,on=self.id_col)
                .merge(traj,on=self.id_col)
                .merge(distance_metrics,on=self.id_col)
                .merge(max_spatial_spread,on=self.id_col)
                .merge(curvature,on=self.id_col)
                #.merge(stop,on=self.id_col)
                .merge(ship_meta[[self.id_col,'shiptype_label']],on=self.id_col,how='left')
                )
