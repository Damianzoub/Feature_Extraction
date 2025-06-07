"""
Γράφω έναν αποδοτικό τρόπο για τον υπολογισμό του.
Δοθέντος ενός συνόλου σημείων P={(xi, yi):i=1,...n}, μας ενδιαφέρει να βρούμε την απόσταση εκείνου του ζεύγους
των σημείων, τα οποία έχουν μέγιστη απόσταση μεταξύ τους, εντός του συνόλου P.
Ο πιο άμεσως τρόπος είναι να υπολογίσουμε το disstance_matrix του συνόλου P με τη βοήθεια της scikitlearn
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) που θα μας δώσει έναν
nxn πίνακα με όλες τις ανά δύο αποστάσεις του συνόλου P. Στη συνέχεια αρκεί απλά εμείς να πάρουμε το μέγιστο
στοιχείου αυτού του πίνακα.

Ο παραπάνω τρόπος, αν και άμεσως, δεν είναι αποδοτικός αφού για 1,000 σημεία ο distance_matrix θα είναι 1,000^2=10^6,
κι ως εκ τούτου είναι υπολογιστικά υπερβολικά κοστοβόρος. Ως εκ τούτου, μπορούμε να υπολογίσουμε πρώτα το Convex Hulll
του συνόλου P. Το CH είναι το κυρτό σύνορο του P, οπότε τα ζεύγη σημείων που ορίζουν τη μέγιστη απόσταση εντός του P,
θα βρίσκονται πάνω του. Στη συνέχεια υπολογίσουμε πάλι το distance_matrix όπως και πριν, μόνο όμως για τα σημεία του CH.
Με αυτό τον τρόπο "ρίχνουμε" την ανάγκη των πολλών υπολογισμών του distance_matrix που απαιτείται με τον ευθύ τρόπο.

Επειδή μας ενδιαφέρουν αποστάσεις πάνω στη σφαίρα (επειδή παίρνουμε τροχιές), θα ορίσουμε custom συνάρτηση απόστασης
ούτως ώστε να χρησιμοποιήσουμε τη Haversine μετρική.

Γράφω τον κώδικα που χρειαζόμαστε και προσάρμοσέ τον όπως εσύ θεωρείς καλύτερα με βάση και τα υπόλοιπα που έχεις
υλοποιήσει.
"""
from sklearn.metrics import pairwise_distances as pwd
from geopy.distance import geodesic
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull , QhullError

#Haversine formula
def haversine(x, y):
    return geodesic(x, y).meters

#implementation of the example in the Data CSV
def calculate_max_spread_per_group(group):
    x = group[['lat','lon']].values

    if len(x)< 3:
        max_spread = 0.0
    else:
        try:
            hull = ConvexHull(x)
            hull_points = x[hull.vertices]
            distance_matrix = pwd(hull_points,hull_points,metric=haversine)
            max_spread= distance_matrix.max()
        except QhullError:
            max_spread = 0.0
    return pd.Series({
        'max_spatial_spread':max_spread
    }) 

#Return the results with (id_col and the max_spatial_spread)
def compute_max_spatial_spread(df,id_col,time_col,lat_col,lon_col):
    required_cols = [id_col,time_col, lat_col,lon_col]
    if not all (cols in df.columns for cols in required_cols):
        raise ValueError("Missing columns")
    
    new_df = df.sort_values(by=[id_col,time_col])
    result = new_df.groupby(id_col).apply(
        lambda group: calculate_max_spread_per_group(group)
    ).reset_index()
    return result[[id_col,'max_spatial_spread']]

#simple example 
def max_spread(df, ch=False):
    """
    This method returns the maximum spread distance for a set of spatial points P max_{i,j}d(P_i, P_j), \forall P_i, P_j \in P
    :param df: A DataFrame containing spatial geolocation of the trajectory (lat, lon).
    :type df: pd.DataFrame
    :param ch: Whether to use Convex Hull or not.
    :type ch: bool
    :return: The maximum distance for the given set of points.
    :rtype: float
    """

    x = df[["lat", "lon"]].values

    # Compute Convex Hull
    hull = ConvexHull(x)
    # Find the points that define the hull
    hull_points = x[hull.vertices]

    # Compute distance_matrix
    distance_matrix = pwd(hull_points, hull_points, metric=haversine)

    return distance_matrix.max()

# test
data = pd.read_csv(r"C:\Users\user\Documents\Feature_Extraction\Feature_Extraction\ais.csv")
data = data.loc[data.shipid == data.shipid.iloc[0]]

max_spread(data)




















