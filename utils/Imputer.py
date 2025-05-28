from sklearn.impute import SimpleImputer
import pandas as pd 
from scipy.interpolate import CubicSpline
import numpy as np


def transform_dataset(data,numeric_columns,categoriclal_columns):
    if numeric_columns != None:
        numerical_imputer = SimpleImputer(strategy='mean')
        data[numeric_columns] = numerical_imputer.fit_transform(data[numeric_columns])
    if categoriclal_columns != None:
        categoriclal_imputer = SimpleImputer(strategy='most_frequent')
        data[categoriclal_columns] = categoriclal_imputer.fit_transform(data[categoriclal_columns])
    return data


def cubic_spline(trajectory, n=1000):
    """
    This method imputes a trajectory with cubic splines, in order to interpolate a smooth representation of it.
    :param trajectory: The trajectory to be imputed.
    :type trajectory: pd.DataFrame
    :param n: Number of total timestamps.
    :type n: int
    :return: An interpolated trajectory.
    :rtype: pd.DataFrame
    """

    tr_shape = trajectory.shape[0]
    t_start = trajectory.index[0]
    t_end = trajectory.index[-1]

    t = np.linspace(0, 1, tr_shape)
    t_new = np.linspace(0, 1, n)
    lat = trajectory["lat"]
    lon = trajectory["lon"]

    cs_lat = CubicSpline(t, lat)
    cs_lon = CubicSpline(t, lon)
    lat_interp = cs_lat(t_new)
    lon_interp = cs_lon(t_new)

    return pd.DataFrame.from_dict({"t": pd.date_range(start=t_start, end=t_end, periods=n), "lat": lat_interp, "lon": lon_interp})
