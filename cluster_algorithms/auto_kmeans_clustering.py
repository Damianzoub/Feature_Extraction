import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import os
from time import sleep

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def automate_kmeans_crosstab(df, features_col, target_col):
    df_copy = df.copy()

    # Keep original target for later use
    target = df_copy[target_col]

    # Clean feature columns
    for col in features_col:
        try:
            df_copy[col] = df_copy[col].astype(float)
        except ValueError:
            print(f"Couldn't convert {col} to float. Dropping it.")
            features_col.remove(col)

    # Prepare feature data
    X = df_copy[features_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Elbow Method
    sse = []
    sse_pca = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        sse.append(kmeans.fit(X_scaled).inertia_)
        kmeans_pca = KMeans(n_clusters=i, random_state=42)
        sse_pca.append(kmeans_pca.fit(X_pca).inertia_)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(range(1, 10), sse, marker='o')
    ax[0].set_title('Elbow Method (original data)')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('SSE')

    ax[1].plot(range(1, 10), sse_pca, marker='o')
    ax[1].set_title('Elbow Method (PCA-reduced data)')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('SSE')

    plt.tight_layout()
    plt.show()

    # Choose data space
    choice = input('Enter which data to use for clustering [1: original, 2: PCA]: ')
    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        sleep(2)
        clear_screen()
        choice = input('Enter which data to use for clustering [1: original, 2: PCA]: ')

    # Choose number of clusters
    while True:
        try:
            n_clusters = int(input('Enter number of clusters (1-9): '))
            if 1 <= n_clusters <= 9:
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid number. Enter an integer from 1 to 9.")
            sleep(2)
            clear_screen()

    # Run KMeans
    if choice == '1':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_pca)

    # Add cluster labels
    df_copy['clusters'] = labels
    df_copy[target_col] = target

    # Crosstab
    crosstab = pd.crosstab(df_copy['clusters'], df_copy[target_col])
    print("\nCluster vs Target Crosstab:")
    print(crosstab)

    return crosstab
