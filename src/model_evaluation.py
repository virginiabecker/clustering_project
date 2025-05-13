import hdbscan
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def print_evaluation_metrics(n_clusters, silhouette_avg, davies_bouldin):
    print(f"\nPara {n_clusters} clusters:")
    print(f"Silhouette score: {silhouette_avg:.4f}")
    print(f"Índice de davies-bouldin: {davies_bouldin:.4f}")

def plot_elbow_method(df, range_n_clusters):
    numeric_df = df.select_dtypes(include=np.number)
    inertias = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(numeric_df)
        inertias.append(kmeans.inertia_)
        print(f"Para {n_clusters} clusters - Inércia: {kmeans.inertia_:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, inertias, marker='o')
    plt.title('Método do cotovelo para determinar o número de clusters')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.xticks(range_n_clusters)
    plt.grid(True)
    plt.savefig('reports/clusters/elbow_method.png')
    plt.show()

def evaluate_dbscan_parameters(df_processed, eps_values, min_samples_values):
    results = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df_processed)
            n_clusters = len(np.unique(labels)) - 1
            n_noise = np.sum(labels == -1)
            silhouette_avg = -1
            if n_clusters > 1 and n_clusters < len(df_processed):
                silhouette_avg = silhouette_score(df_processed[labels != -1], labels[labels != -1])

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_avg': silhouette_avg
            })
    return pd.DataFrame(results).sort_values(by='silhouette_avg', ascending=False)

def evaluate_hdbscan_parameters(df_processed, min_cluster_size_values, min_samples_values=None):
    results = []
    for mcs in min_cluster_size_values:
        if min_samples_values is None:
            ms_values = [mcs]
        else:
            ms_values = min_samples_values

        for ms in ms_values:
            try:
                hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                labels = hdbscan_model.fit_predict(df_processed)
                n_clusters = len(np.unique(labels))
                n_noise = np.sum(labels == -1)
                silhouette_avg = -1
                if n_clusters > 1 and n_clusters < len(df_processed[labels != -1]):
                    silhouette_avg = silhouette_score(df_processed[labels != -1], labels[labels != -1])
                results.append({
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_avg': silhouette_avg
                })
            except Exception as e:
                print(f"Erro ao rodar HDBSCAN com mcs={mcs}, ms={ms}: {e} ")
                results.append({
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'n_clusters': -99,
                    'n_noise': -99,
                    'silhouette_avg': -99
                })

    return pd.DataFrame(results).sort_values(by='silhouette_avg', ascending=False)
      
  
def plot_k_nearest_neighbors_eps(df_processed, k=4):
    start_time = time.time()
    neighbors = NearestNeighbors(n_neighbors=k)
    print(f"Tempo após inicializar NearestNeighbors: {time.time() - start_time:.2f} segundos")   

    neighbors_fit = neighbors.fit(df_processed)
    print(f"Tempo após fit: {time.time() - start_time:.2f} segundos") 
    distances, indices = neighbors_fit.kneighbors(df_processed)
    print(f"Tempo após kneighbors: {time.time() - start_time:.2f} segundos")

    distances = np.sort(distances, axis=0)
    print(f"Tempo após sort: {time.time() - start_time:.2f} segundos")

    distances = distances[:, k-1]
    print(f"Tempo após slicing: {time.time() - start_time:.2f} segundos")

    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.xlabel(f"Pontos ordenados pela distância ao {k}ésimo vizinho")
    plt.ylabel(f"Distância ao {k}ésimo vizinho")
    plt.title(f"Gráfico do {k} vizinho mais próximo para estimativa de eps (k={k})")
    plt.grid(True)
    plt.savefig('reports/clusters/knn_eps_estimation.png')
    plt.show()

