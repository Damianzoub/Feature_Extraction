from sklearn.impute import SimpleImputer
import pandas as pd 

def transform_dataset(data,numeric_columns,categoriclal_columns):
    if numeric_columns != None:
        numerical_imputer = SimpleImputer(strategy='mean')
        data[numeric_columns] = numerical_imputer.fit_transform(data[numeric_columns])
    if categoriclal_columns != None:
        categoriclal_imputer = SimpleImputer(strategy='most_frequent')
        data[categoriclal_columns] = categoriclal_imputer.fit_transform(data[categoriclal_columns])
    return data