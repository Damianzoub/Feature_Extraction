import pandas as pd
import datetime as dt
from sklearn.impute import SimpleImputer
import numpy as np
from geopy.distance import geodesic
'''
features for extraction:
1) ROT: mean value , std
   Avg Speed: max, mean ,min , std
   Accelearation: mean,max,min,std
   Trajectories: start_latitude, start_longitude, end_latitude, end_longitude , start_time , end_time, duration_seconds, duration_hours
'''

class DataTransformer:
    def __init__(self,dataset_path,time_col='t',id_col='shipid',
                 speed_col='speed',heading_col='heading',lat_col='lat',
                 long_col='long',course_col='course',shiptype_col='shiptype'
                 ,destination_col='destination',numeric_columns=None,categorical_columns=None):
        self.dataset_path = dataset_path
        self.data = None
        self.time_col = time_col
        self.id_col = id_col
        self.speed_col = speed_col
        self.heading_col = heading_col
        self.lat_col = lat_col
        self.long_col = long_col
        self.course_col = course_col
        self.destination_col = destination_col
        self.shiptype_col =shiptype_col
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
    
    def load_data(self):
        if self.dataset_path.endswith('.csv'):
            self.data = pd.read_csv(self.dataset_path)
            #return the dataset
            
        else:
            raise ValueError('Only CSV files are supported for now')

    """if we don't have null values then this function
      below is not needed"""
    def transform_dataset(self):
        
        if self.data is None:
             raise ValueError("No data loaded. call first load_data() method")

        numeric_col = [col for col in self.numeric_columns if col in self.data.columns]
        cat_col = [col for col in self.categorical_columns if col in self.data.columns]

        print("Numeric columns to impute:", numeric_col)
        print("Categorical columns to impute: ",cat_col)
        
        numericImputer = SimpleImputer(strategy='mean')
        categoricalImputer = SimpleImputer(strategy='most_frequent')

        if numeric_col:
             self.data[numeric_col] = numericImputer.fit_transform(self.data[numeric_col])
        if cat_col:
             self.data[cat_col] = categoricalImputer.fit_transform(self.data[cat_col])

    def exist_null(self):
         exist_null = []
         for col in self.data.columns:
              if self.data[col].isnull().sum() > 0:
                 exist_null.append((col,self.data[col].isnull().sum()))
         
         if len(exist_null) == 0 :
              return None 
         else:
              return exist_null   

    def _categorize_time(self):
         self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
         # Fix rare edge case where timestamps end with 59:59 to avoid time bucketing conflicts
         self.data.loc[(self.data[self.time_col].dt.minute==59) & (self.data[self.time_col].dt.second==59), self.time_col] = self.data[self.time_col] + pd.Timedelta(seconds=1)
        
         
            
    #returns the statistical values about the speed
    def average_speed_per_id(self):
            if self.speed_col not in self.data.columns:
                 raise ValueError(f"{self.speed_col} doesn't exist in the dataset")
            
            self._categorize_time()

            #transforms the data time to datime for categorization later
            new_df = self.data
            
            #groups the id of an element for every timestamp that this id is located 
            #and calculates the statical values of this ID
            aggregate = new_df.groupby([self.id_col,self.time_col])[self.speed_col].agg(
                 avg_speed = 'mean',
                 max_speed= 'max',
                 min_speed= 'min',
                 std_speed= 'std'
            ).reset_index()

            #creates the total average statistical values (max,mean,min,std) for speed
            return aggregate.groupby(self.id_col).agg({
                 "avg_speed":"mean",
                 "max_speed":"max",
                 "min_speed":'min',
                 "std_speed":"std"
            }).reset_index()
    
    #acceleration function per id 
    # statistical values (mean,max,min,std) per id
    def acceleration_per_id(self):
        if self.speed_col not in self.data.columns:
             raise ValueError(f"{self.speed_col} column doesn't exist")
        self._categorize_time()
        new_df = self.data

        new_df = new_df.sort_values(by=[self.id_col,self.time_col])
        new_df['time_diff'] = new_df.groupby(self.id_col)[self.time_col].diff().dt.total_seconds().fillna(1)
        new_df['speed_diff'] = new_df.groupby(self.id_col)[self.speed_col].diff().fillna(0)
        new_df['acceleration'] = new_df['speed_diff']/new_df['time_diff']
        return new_df.groupby(self.id_col)['acceleration'].agg(
             avg_acceleration = 'mean',
             max_acceleration='max',
             min_acceleration = 'min',
             std_acceleration = 'std'
        ).reset_index()

    #ROT statistical values per id
    #mean, std values
    def rot_per_id(self):
         if self.heading_col not in self.data.columns:
              raise ValueError(f"{self.heading_col} column doesn't exist")
        
         self._categorize_time()
         new_df = self.data 
         
         new_df = new_df.sort_values(by=[self.id_col,self.time_col])
         new_df['heading_diff'] = new_df.groupby(self.id_col)[self.heading_col].diff()
         new_df['time_diff'] = new_df.groupby(self.id_col)[self.time_col].diff().dt.total_seconds().fillna(1)
         new_df['ROT'] = new_df['heading_diff']/new_df['time_diff']
         return new_df.groupby(self.id_col)['ROT'].agg(
              rot_mean = 'mean',
              rot_std = 'std'
         ).reset_index()
    
    def _compute_total_and_straightness_metrics(self):
        required_cols = [self.id_col, self.time_col, self.lat_col, self.long_col]
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError("Missing one or more required columns in the dataset.")

        new_df = self.data.sort_values(by=[self.id_col, self.time_col])

        result = new_df.groupby(self.id_col).apply(
            lambda group: self._compute_metrics(group)
        ).reset_index()

        return result[[self.id_col, 'total_distance_km', 'straightness_ratio', 'tortuosity']]

    def _compute_metrics(self, group):
        coords = list(zip(group[self.lat_col], group[self.long_col]))
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
    #we combine all the methods of this script
    #so we can get the DataFrame of the features
    def get_all_features(self):
         speed = self.average_speed_per_id()
         acceleration = self.acceleration_per_id()
         rot = self.rot_per_id()
         traj = self.trajectory()
         spatial_metrics = self._compute_total_and_straightness_metrics()
         return (speed
                .merge(acceleration, on=self.id_col)
                .merge(rot, on=self.id_col)
                .merge(traj,on=self.id_col)
                .merge(spatial_metrics,on=self.id_col)
                )

    def trajectory(self):
         if self.long_col not in self.data.columns:
              raise ValueError(f'{self.long_col} not in the Dataset Columns')
         if self.lat_col not in self.data.columns:
              raise ValueError(f'{self.lat_col} not in the Dataset column')
         new_data = self.data
         new_data['t'] = pd.to_datetime(self.data['t'])
     
         grouped  = new_data.sort_values([self.id_col,self.time_col]).groupby(self.id_col)
         features = grouped.agg(
              start_lat = (self.lat_col,'first'),
              start_long = (self.long_col,'first'),
              end_lat = (self.lat_col,'last'),
              end_long = (self.long_col,'last'),
              start_time = (self.time_col,'first'),
              end_time = (self.time_col , 'last')
         )
         features['duration_second'] = (features['end_time']-features['start_time']).dt.total_seconds()
         features['duration_hour'] = features['duration_second']/3600
         features = features.reset_index()
         return features[[self.id_col,'start_lat','start_long','end_lat',
                          'end_long','start_time','end_time','duration_second'
                          ,'duration_hour']]

#examples below if you want to test it remove the ""

"""
data_transform = DataTransformer(
        dataset_path='ais.csv',
        time_col='t',
        id_col='shipid',
        speed_col='speed',
        heading_col='heading',
        lat_col='lat',
        long_col='lon',
        course_col='course',
        shiptype_col='shiptype',
        destination_col='destination',
        numeric_columns=['heading', 'course', 'speed'],
        categorical_columns=['shiptype', 'destination']
    )

data_transform.load_data()
data_transform.transform_dataset()
acceleration_df = data_transform.acceleration_per_id()
rot_df = data_transform.rot_per_id()
speed_df = data_transform.average_speed_per_id()
features_df = data_transform.get_all_features()
print(features_df)

"""


