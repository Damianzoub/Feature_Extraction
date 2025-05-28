import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt


def speed(df):
    df['t'] = pd.to_datetime(df['t'])
    speeds = [0]
    for i in range(1, df.shape[0]):
        coor_pr = (df.iloc[i-1]["lat"], df.iloc[i-1]["lon"])
        coor_next = (df.iloc[i]["lat"], df.iloc[i]["lon"])
        t_pr = df.iloc[i-1]["t"]
        t_next = df.iloc[i]["t"]

        distance = geodesic(coor_pr, coor_next).meters
        dt = (t_next - t_pr).total_seconds()
        if dt == 0:
            speeds.append(0)
        else:
            speeds.append(distance / dt)

    return pd.DataFrame.from_dict({"t": df["t"], "Speed": speeds})
