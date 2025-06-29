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
print(features_df)
excluded_cols = ['shipid','start_time','end_time','start_hour','start_minute','end_hour','std_speed','end_minute','zigzag_index','shiptype']
label_cols = ['fishing','cargo']
filtered_df = features_df[features_df['shiptype'].isin(label_cols)].copy()
feature_cols = [col for col in features_df.columns if col not in excluded_cols]
results = ClassifierPipeline(filtered_df,feature_cols,'shiptype')

#print(results.execute())
#clustering_result = ClusteringPipeline(filtered_df,feature_cols,'shiptype',len(label_cols))
#ars ,nmis = clustering_result.cluster_dbscan()
