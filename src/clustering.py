from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
import numpy as np

def run_kmeans_clustering(df_processed, n_clusters=3, random_state=42):
    numeric_df = df_processed.select_dtypes(include=['int64', 'float64', 'uint8'])
    print("\nColunas numéricas:")
    print(numeric_df.columns)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(numeric_df)
    df_with_clusters = numeric_df.copy()
    df_with_clusters['Cluster'] = kmeans.labels_
    return df_with_clusters

def run_dbscan_clustering(df_processed, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_processed)
    df_with_clusters = df_processed.copy()
    df_with_clusters['DBSCAN_Cluster'] = labels
    return df_with_clusters

def run_hdbscan_clustering(df_processed, min_cluster_size=5, min_samples=None):
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = hdbscan_model.fit_predict(df_processed)
    df_with_clusters = df_processed.copy()
    df_with_clusters['HDBSCAN_Cluster'] = clusters
    return df_with_clusters
