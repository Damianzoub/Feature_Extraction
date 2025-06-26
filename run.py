from DataTransform import DataTransformer
from utils.shiptype_map import map_shiptype
from ClassificationPipe.Classifier_Pipeline import ClassifierPipeline
from cluster_algorithms.ClusteringPipe import ClusteringPipeline
import  numpy as np 
dt = DataTransformer(
    dataset_path="ais.csv",
    time_col="t",
    id_col="shipid",
    lat_col="lat",
    lon_col="lon",
    shiptype_col="shiptype",
    numeric_cols=['heading','course','speed'],
    categorical_cols=['shiptype','destination']
)

dt.load_data()
dt.transform_dataset()
features_df = dt.extract_features(mode='all')
#print(features_df.isnull().sum())
inf_pos_mask = features_df == np.inf
inf_pos_counts = inf_pos_mask.sum()

# Check for -inf values
inf_neg_mask = features_df == -np.inf
inf_neg_counts = inf_neg_mask.sum()

# Combine the two
total_inf_counts = inf_pos_counts + inf_neg_counts

# Show only the columns that actually have inf values
inf_columns = total_inf_counts[total_inf_counts > 0]


print(inf_columns)
#excluded_cols = ['shipid','start_time','end_time','start_hour','start_minute','end_hour','std_speed','end_minute','zigzag_index','shiptype']
#label_cols = ['fishing','cargo']
#filtered_df = features_df[features_df['shiptype'].isin(label_cols)].copy()
#feature_cols = [col for col in features_df.columns if col not in excluded_cols]
#results = ClassifierPipeline(filtered_df,feature_cols,'shiptype')

#print(results.execute())
#clustering_result = ClusteringPipeline(filtered_df,feature_cols,'shiptype',len(label_cols))
#ars ,nmis = clustering_result.cluster_dbscan()

"""
shipid', 'avg_speed', 'max_speed', 'min_speed', 'std_speed',
       'avg_acceleration', 'max_acceleration', 'min_acceleration',
       'std_acceleration', 'rot_mean', 'rot_std', 'start_lat', 'start_long',
       'end_lat', 'end_long', 'start_time', 'end_time', 'duration_second',
       'start_year', 'start_month', 'start_day', 'start_hour', 'start_minute',
       'end_year', 'end_month', 'end_day', 'end_hour', 'end_minute',
       'total_distance_km', 'straightness_ratio', 'tortuosity',
       'max_spatial_spread', 'max_curvature', 'min_curvature',
       'mean_curvature', 'std_curvature', 'median_curvature', 'zigzag_index',
       'shiptype_label'],
       

       """
