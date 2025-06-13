import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline



def curvature_results(df,id_col,time_col,lat_col,lon_col):
      required_columns = [id_col,lat_col,lon_col]
      if not all(cols in df.columns for cols in required_columns):
             raise ValueError("Missing Columns")
    
      result = df.groupby(id_col).apply(
          lambda group: curvature_calculation(group,time_col,lat_col,lon_col)
      ).reset_index()
      return result[[id_col,"max_curvature","min_curvature","mean_curvature","std_curvature","median_curvature"]]

def curvature_calculation(group,time_col,lat_col,lon_col,n=100_000):

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


def curvature(df, n=100000):
    """
    This method computes basic statistical features as extracted by curvature.
    :param df: A DataFrame containing geospatial information regarding the trajectory.
    :type df: pd.DataFrame
    :param n: The total number of points used after imputation to approximate a smooth version of the trajectory.
    :type n: int
    :return: Basic statistical features of curvature.
    :rtype: dict
    """

    # Transform time in [0, 1]
    t = np.linspace(0, 1, df.shape[0])

    # Fit cubic splines for latitude and longitude
    lat_spline = CubicSpline(t, df["lat"])
    lon_spline = CubicSpline(t, df["lon"])

    # Fine time interval
    t_fine = np.linspace(0, 1, n)

    # Approximate derivatives over the fined time
    dlat = lat_spline(t_fine, 1)
    dlon = lon_spline(t_fine, 1)
    ddlat = lat_spline(t_fine, 2)
    ddlon = lon_spline(t_fine, 2)

    # Compute curvature over the fined time
    curv = (dlat * ddlon - dlon * ddlat) / (dlat**2 + dlon**2)**1.5

    return {"max_curvature": curv.max(),
            "min_curvature": curv.min(),
            "mean_curvature": curv.mean(),
            "std_curvature": curv.std(),
            "median_curvature": np.median(curv)}
