import pandas as pd
import matplotlib.pyplot as plt 
from kneed import KneeLocator 
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score ,normalized_mutual_info_score,silhouette_score
from scipy.optimize import linear_sum_assignment
import numpy as np

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

        

        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        correct_counts_df = self.count_correct_clusters_per_label(true_labels, pred_labels)
        print(correct_counts_df)
        return np.round(ari,2),np.round(nmi,2), correct_counts_df


    def count_correct_clusters_per_label(self, true_labels, pred_labels, label_mapping=None):
        """
        Counts the number of correctly clustered samples per true label 
        by finding the best matching between cluster IDs and true labels.
        
        Parameters:
            true_labels: array-like of true class labels (can be encoded integers or strings).
            pred_labels: array-like of predicted cluster labels.
            label_mapping: dict or None. Optional mapping from encoded labels to original string labels.
                        For example: {0: 'cargo', 1: 'fishing'}
        
        Returns:
            pandas DataFrame with columns ['ShipType', 'CorrectClusterCount', 'TotalSamples'] where
            ShipType contains original string labels.
        """
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        unique_true = np.unique(true_labels)
        unique_pred = np.unique(pred_labels)

        # Build confusion matrix: rows=true labels, cols=pred clusters
        confusion = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
        for i, t in enumerate(unique_true):
            for j, p in enumerate(unique_pred):
                confusion[i, j] = np.sum((true_labels == t) & (pred_labels == p))

        # Find the best assignment to maximize matches
        row_ind, col_ind = linear_sum_assignment(-confusion)

        # Prepare results with original label names if mapping provided
        results = []
        for r, c in zip(row_ind, col_ind):
            label_name = label_mapping[unique_true[r]] if label_mapping is not None else unique_true[r]
            results.append({
                'ShipType': label_name,
                'CorrectClusterCount': confusion[r, c],
                'TotalSamples': np.sum(true_labels == unique_true[r])
            })

        return pd.DataFrame(results)
        
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
        correct_counts_df = self.count_correct_clusters_per_label(true_labels, pred_labels)
        print(correct_counts_df)
        return (np.round(adjusted_rand_score(true_labels,pred_labels),2), np.round(normalized_mutual_info_score(true_labels,pred_labels),2),correct_counts_df)
    
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


    


    
