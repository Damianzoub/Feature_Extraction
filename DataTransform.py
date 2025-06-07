import pandas as pd
#file path utils for 
from utils.data_loader import load_csv
from utils.Imputer import transform_dataset
from utils.time_utils import categorize_time
from features.speed import average_speed_per_id
from features.acceleration import acceleration_per_id
from features.rot import rot_per_id
from features.trajectory import trajectory
from features.distance_and_straightness import _compute_total_and_straightness_metrics
from features.max_spatial_spread import compute_max_spatial_spread
from features.curvature import curvature_results


class DataTransformer:
    def __init__(self,dataset_path,time_col='t',id_col='shipid',speed_col='speed',
                 heading_col='heading',lat_col='lat',lon_col='lon',course_col='course'
                 ,shiptype_col='shiptype',destination_col='destination',numeric_cols=None,categorical_cols=None):
        
        self.dataset_path= dataset_path
        self.data = None
        self.time_col=time_col
        self.id_col=id_col
        self.speed_col=speed_col
        self.heading_col=heading_col
        self.lat_col=lat_col
        self.lon_col=lon_col
        self.course_col=course_col
        self.shiptype_col=shiptype_col
        self.destination_col=destination_col
        self.numeric_cols = numeric_cols
        self.categorical_cols=categorical_cols

    def load_data(self):
        self.data = load_csv(self.dataset_path)
    
    def transfrom_dataset(self):
        if self.data is None:
            raise ValueError('No data loaded')
        self.data = transform_dataset(self.data,numeric_columns=self.numeric_cols,categoriclal_columns=self.categorical_cols)
    
    def exist_null(self):
        return [(col,self.data[col].isnull().sum()) for col in self.data.columns if self.data[col].isnull().sum() >0 ] or None

    def get_all_features(self):
        speed = average_speed_per_id(self.data,self.id_col,self.time_col,self.speed_col)
        acceleration= acceleration_per_id(self.data,self.time_col,self.id_col,self.speed_col)
        rot = rot_per_id(self.data,self.heading_col,self.id_col,self.time_col)
        traj = trajectory(self.data,self.id_col,self.time_col,self.lat_col,self.lon_col)
        distance_metrics = _compute_total_and_straightness_metrics(self.data,self.id_col,self.time_col,self.lat_col,self.lon_col)
        max_spread = compute_max_spatial_spread(self.data,self.id_col,self.time_col,self.lat_col,self.lon_col)
        curvature = curvature_results(self.data,self.id_col,self.time_col,self.lat_col,self.lon_col)


        return (speed.merge(acceleration,on=self.id_col)
                .merge(rot,on=self.id_col)
                .merge(traj,on=self.id_col)
                .merge(distance_metrics,on=self.id_col)
                .merge(max_spread,on=self.id_col)
                .merge(curvature,on=self.id_col)
                )
        
