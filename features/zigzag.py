import pandas as pd 
import numpy as np
from utils.time_utils import categorize_time
def calculate_theta(lat1,lon1,lat2,lon2):
    dLon = np.radians(lon2-lon1)
    lat1= np.radians(lat1)
    lat2 = np.radians(lat2)

    #preapring for calculation of theta
    x = np.sin(dLon)*np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dLon)

    #theta calculation
    theta = np.arctan2(x,y)
    #normalization of theta in the range [-pi,pi]
    theta = (np.degrees(bearing)+360)%360
    return theta

#angle_threshold is in degrees
def zigzag_index(lat,lon,angle_threshold=10):
    if len(lat)< 3:
        return 0
    
    bearigs = [calculate_theta(lat[i],lon[i],lat[i+1],lon[i+1]) for i in range(len(lat)-1)]
    #calculating the differences in theta
    dtheta = np.diff(bearigs)
    dtheta = (dtheta+np.pi)%(2*np.pi) - np.pi 
    #transforming degress to rads for the delta function
    angle_threshold = np.deg2rad(angle_threshold)
    zigzag_count = np.sum(np.abs(dtheta)>angle_threshold)
    
    return int(zigzag_count)

def compute_zigzag(df,shipid_col,time_col,lat_col,lon_col):
    new_df = categorize_time(df,time_col)
    new_df = new_df.sort_values(by=[shipid_col,time_col])

    results = []
    for shipID , group in new_df.groupby(shipid_col):
        lat = group[lat_col].values
        lon = group[lon_col].values
        count = zigzag_index(lat,lon)
        results.append({
            "shipid":shipID,
            "count_zigzag":count
        })
    return pd.DataFrame(results)
    
