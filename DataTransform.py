import pandas as pd
#file path utils for 
from utils.data_loader import load_csv
from utils.Imputer import transform_dataset
from utils.cache_utils import save_cache,load_cache
from feature_pipeline.feature_pipeline import FeaturePipeline
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

    def load_data(self):
        self.data = load_csv(self.dataset_path)
    
    def transform_dataset(self):
        if self.data is None:
            raise ValueError('No data loaded')
        self.data = transform_dataset(self.data,numeric_columns=self.numeric_cols,categoriclal_columns=self.categorical_cols)
    
    def exist_null(self):
        return [(col,self.data[col].isnull().sum()) for col in self.data.columns if self.data[col].isnull().sum() >0 ] or None

    

    def extract_features(self,mode='all'):
        """
        mode: 'all' | 'statistical' | 'per_se'
        Returns the selected feature set.
        """
        feature_pipeline = FeaturePipeline(**self.col_kwargs)
        if mode == "statistical":
            return feature_pipeline.statistical_measures(self.data)
        elif mode == "per_se":
            return feature_pipeline.features_per_se(self.data)
        elif mode == "all":
            return feature_pipeline.extract_all(self.data)
        else:
            raise ValueError("Invalid mode. Choose from: 'all', 'statistical', or 'per_se'")
        
    
    
    
        
