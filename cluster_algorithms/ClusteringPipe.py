import pandas as pd
import matplotlib.pyplot as plt 
from kneed import KneeLocator 
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score ,normalized_mutual_info_score,silhouette_score

class ClusteringPipeline:
    def __init__(self,features_df,features_columns,target_label,number_clusters):
        self.features_df = features_df.copy()
        self.features_columns = features_columns
        self.target_label = target_label
        self.number_clusters = number_clusters

    def cluster_dbscan(self,eps=0.5,min_samples=5):
        true_labels = self.label_Encoding()
        X_Scaled = self.preprocess_data()

        dbscan = DBSCAN(eps=eps,min_samples=min_samples)
        pred_labels = dbscan.fit_predict(X_Scaled)
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = list(pred_labels).count(-1)

        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise}")
        print("True Labels (first 10):", true_labels[:10])
        print("Predicted Labels (first 10):", pred_labels[:10])

        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        print("Adjusted Rand Index (ARI):", ari)
        print("Normalized Mutual Info (NMI):", nmi)
        return ari,nmi

    def show_inertia(self,X):
        inertia = []
        for i in range(1,10):
            kmeans = KMeans(n_clusters=i,random_state=42)
            inertia.append(kmeans.fit(X).inertia_)
        plt.grid(True)
        plt.plot(range(1,10),inertia)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Rule")
        return inertia
    
    def label_Encoding(self):
        le = LabelEncoder()
        return le.fit_transform(self.features_df[self.target_label])
        

    def preprocess_data(self):
        X = self.features_df[self.features_columns]
        X_Scaled = StandardScaler().fit_transform(X)
        return X_Scaled
    
    def cluster(self):
        true_labels = self.label_Encoding()
        X_Scaled = self.preprocess_data()
        

        kmeans = KMeans(n_clusters=self.number_clusters,random_state=42)
        pred_labels = kmeans.fit_predict(X_Scaled)
        print("True Cluster Labels:",true_labels[:10])
        print("Predicted Cluster Labels:", pred_labels[:10])
        return (adjusted_rand_score(true_labels,pred_labels), normalized_mutual_info_score(true_labels,pred_labels))
    
    def result(self,kmeans,pred_labels,X_Scaled):
        centers = kmeans.cluster_centers_
        plt.grid(True)
        plt.scatter(
            centers[:,0],centers[:,1],
            c='red',
            s=150,
            label="Centroids"
        )

        plt.scatter(
            X_Scaled[:,0],X_Scaled[:,1],
            c=pred_labels,
            cmap='viridis',
            label="Data Points"
        )
        plt.title("Cluster Visualization")
        plt.legend()
        plt.show()


    


    
