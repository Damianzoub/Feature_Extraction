import pandas as pd
import datetime as dt
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances as pwd
from scipy.spatial import ConvexHull, QhullError
import numpy as np
from geopy.distance import geodesic
from scipy.interpolate import CubicSpline
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
                 lon_col='lon',course_col='course',shiptype_col='shiptype'
                 ,destination_col='destination',numeric_columns=None,categorical_columns=None):
        self.dataset_path = dataset_path
        self.data = None
        self.time_col = time_col
        self.id_col = id_col
        self.speed_col = speed_col
        self.heading_col = heading_col
        self.lat_col = lat_col
        self.lon_col = lon_col
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

   #Total Distance and Tortuosity and Straightness_Ratio
    def _compute_total_and_straightness_metrics(self):
        required_cols = [self.id_col, self.time_col, self.lat_col, self.lon_col]
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError("Missing one or more required columns in the dataset.")

        new_df = self.data.sort_values(by=[self.id_col, self.time_col])

        result = new_df.groupby(self.id_col).apply(
            lambda group: self._compute_metrics(group)
        ).reset_index()

        return result[[self.id_col, 'total_distance_km', 'straightness_ratio', 'tortuosity']]

    def _compute_metrics(self, group):
        coords = list(zip(group[self.lat_col], group[self.lon_col]))
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
    #Haversine formula from geopy
    def haversine(self,x,y):
          return geodesic(x,y).meters
   
    def compute_max_spatial_spread(self):
         required_cols = [self.id_col,self.time_col,self.lat_col,self.lon_col]
         if not all(col in self.data.columns for col in required_cols):
              raise ValueError("Missing one or more required columns")
         
         new_df = self.data.sort_values(by=[self.id_col,self.time_col])
         result = new_df.groupby(self.id_col).apply(
          lambda group: self.calculate_max_spread_per_group(group)
         ).reset_index()
         return result[[self.id_col,'max_spatial_spread']]

    def calculate_max_spread_per_group(self,df):
      x = df[['lat','lon']].values

      #Not enought points for ConvexHull algorithm
      if len(x) < 3:
         max_spread= 0.0 
      else:
         try:
             hull = ConvexHull(x)
             hull_points = x[hull.vertices]
             distance_matrix = pwd(hull_points,hull_points,metric=self.haversine)
             max_spread= distance_matrix.max()
         except QhullError:
             max_spread= 0.0 #ConvexHull failed     
          
      return pd.Series({
        'max_spatial_spread':max_spread
      })
    



    def curvature_results(self):
      required_columns = [self.id_col,self.lat_col,self.lon_col]
      if not all(cols in self.data.columns for cols in required_columns):
             raise ValueError("Missing Columns")
    
      result = self.data.groupby(self.id_col).apply(
          lambda group: self.curvature_calculation(group,self.time_col,self.lat_col,self.lon_col)
      ).reset_index()
      return result[[self.id_col,"max_curvature","min_curvature","mean_curvature","std_curvature","median_curvature"]]

    def curvature_calculation(self,group,time_col,lat_col,lon_col,n=100_000):

          if group.shape[0] < 3:
               return pd.Series({"max_curvature": 0.0,
                    "min_curvature": 0.0,
                    "mean_curvature": 0.0,
                    "std_curvature": 0.0,
                    "median_curvature": 0.0
                    }) 

          t = group.sort_values(by=time_col)
          t = np.linspace(0,1,group.shape[0])

          t_fine = np.linspace(0,1,n)

          lat_spline = CubicSpline(t, group[lat_col].values)
          lon_spline = CubicSpline(t, group[lon_col].values)

          # Fine time interval
          t_fine = np.linspace(0, 1, n)

          # Approximate derivatives over the fined time
          dlat = lat_spline(t_fine, 1)
          dlon = lon_spline(t_fine, 1)
          ddlat = lat_spline(t_fine, 2)
          ddlon = lon_spline(t_fine, 2)

          # Compute curvature over the fined time
          curv = (dlat * ddlon - dlon * ddlat) / (dlat**2 + dlon**2)**1.5

          return pd.Series({"max_curvature": curv.max(),
                    "min_curvature": curv.min(),
                    "mean_curvature": curv.mean(),
                    "std_curvature": curv.std(),
                    "median_curvature": np.median(curv)
                    })


    #we combine all the methods of this script
    #so we can get the DataFrame of the features
    def get_all_features(self):
         speed = self.average_speed_per_id()
         acceleration = self.acceleration_per_id()
         rot = self.rot_per_id()
         traj = self.trajectory()
         spatial_metrics = self._compute_total_and_straightness_metrics()
         max_spatial_spread = self.compute_max_spatial_spread()
         curvature = self.curvature_results()
         return (speed
                .merge(acceleration, on=self.id_col)
                .merge(rot, on=self.id_col)
                .merge(traj,on=self.id_col)
                .merge(spatial_metrics,on=self.id_col)
                .merge(max_spatial_spread,on=self.id_col)
                .merge(curvature,on=self.id_col)
                )

    def trajectory(self):
         if self.lon_col not in self.data.columns:
              raise ValueError(f'{self.lon_col} not in the Dataset Columns')
         if self.lat_col not in self.data.columns:
              raise ValueError(f'{self.lat_col} not in the Dataset column')
         new_data = self.data
         new_data['t'] = pd.to_datetime(self.data['t'])
     
         grouped  = new_data.sort_values([self.id_col,self.time_col]).groupby(self.id_col)
         features = grouped.agg(
              start_lat = (self.lat_col,'first'),
              start_lon = (self.lon_col,'first'),
              end_lat = (self.lat_col,'last'),
              end_lon = (self.lon_col,'last'),
              start_time = (self.time_col,'first'),
              end_time = (self.time_col , 'last')
         )
         features['duration_second'] = (features['end_time']-features['start_time']).dt.total_seconds()
         features['duration_hour'] = features['duration_second']/3600
         features = features.reset_index()
         return features[[self.id_col,'start_lat','start_lon','end_lat',
                          'end_lon','start_time','end_time','duration_second'
                          ,'duration_hour']]

#examples below if you want to test it remove the ""


data_transform = DataTransformer(
        dataset_path='ais.csv',
        time_col='t',
        id_col='shipid',
        speed_col='speed',
        heading_col='heading',
        lat_col='lat',
        lon_col='lon',
        course_col='course',
        shiptype_col='shiptype',
        destination_col='destination',
        numeric_columns=['heading', 'course', 'speed'],
        categorical_columns=['shiptype', 'destination']
    )

data_transform.load_data()
data_transform.transform_dataset()
features_df = data_transform.get_all_features()
print(features_df)



