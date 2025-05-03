### DataTransformer

**DataTransformer** is a Python class designed to load and preprocess data. It was created for feature engineering utilities including calculations of **speed, acceleration and ROT(rate of turn)**

## Features
- Data Loading: Supports CSV files.
- Missing Value Handling: Imputes numeric and categorical columns.
- Time Normalization: Converts and sanitizes timestamp data.
- Feature Extraction:
    1. Speed Statistics: max,mean,min,std per ID
    2. Acceleration Statistics: mean,max,min,std per ID
    3. Rate Of Turn (ROT): mean and std per ID
    4. Unified Feature Extraction: **get_all_features()** returns a dataset of all extracted features

## Installation 
```bash 
git clone https://github.com/Damianzoub/Feature_Extraction.git
cd Feature_Extraction
pip install -r requirements.txt
```

## Usage

1. Initialize the Transformer
    ```python

    from transformation import DataTransformer

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
    ```
2. Load and Preprocess Data
    ```python
    data_transform.load_data()
    data_transform.transform_dataset()
    ```

3. Extract Features
    ```python
    acceleration_df = data_transform.acceleration_per_id()
    rot_df = data_transform.rot_per_id()
    speed_df = data_transform.average_speed_per_id()
    features_df = data_transform.get_all_features()
    ```
