import pandas as pd 
from features.stops import count_stops

df = pd.read_csv('dummy_ship_trajectories.csv')
df = df.drop('shiptype',axis=1)
results = count_stops(df,'shipid','timestamp','lat','lon')
print(results)
