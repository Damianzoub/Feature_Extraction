from DataTransform import DataTransformer

dt = DataTransformer(
    dataset_path="ais.csv",
    time_col="t",
    id_col="shipid",
    speed_col="speed",
    heading_col="heading",
    lat_col="lat",
    long_col="lon",
    course_col="course",
    shiptype_col="shiptype",
    destination_col="destination",
    numeric_cols=['heading','course','speed'],
    categorical_cols=['shiptype','destination']
)

dt.load_data()
dt.transfrom_dataset()
features_df = dt.get_cached_features(mode='all')
print(features_df)
