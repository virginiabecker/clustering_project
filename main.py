import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.data_loader import load_and_merge_data
from src.exploratory_analysis import show_descriptive_stats, show_value_counts
from src.preprocessing import preprocess_data, StandardScaler
from src.clustering import run_kmeans_clustering, run_dbscan_clustering, run_hdbscan_clustering
from src.visualization import visualize_clusters_pca, visualize_cluster_features_kmeans, visualize_cluster_features_dbscan, visualize_cluster_features_hdbscan
from src.model_evaluation import plot_elbow_method, print_evaluation_metrics, evaluate_dbscan_parameters, evaluate_hdbscan_parameters, plot_k_nearest_neighbors_eps


def find_highly_correlated_features_per_cluster(df_with_clusters, cluster_column, top_n=5):
    unique_clusters = df_with_clusters[cluster_column].unique()
    numeric_cols = df_with_clusters.select_dtypes(include=np.number).columns.tolist()
    if cluster_column in numeric_cols:
        numeric_cols.remove(cluster_column)
    print(f"\n--- Top {top_n} pares de features mais correlacionados por cluster ---")
    for cluster in unique_clusters:
        cluster_df = df_with_clusters[df_with_clusters[cluster_column] == cluster][numeric_cols]
        if len(cluster_df) > 2:
            corr_matrix = cluster_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            stacked_corr = upper.stack().sort_values(ascending=False)
            print(f"\nCluster {cluster}:")
            if not stacked_corr.empty:
                top_corr_pairs = stacked_corr.head(top_n)
                for (col1, col2), corr_value in top_corr_pairs.items():
                    print(f"  Correlação entre {col1} e {col2}: {corr_value:.4f}")
            else:
                print("  Não foi possível calcular correlações significativas (poucos pontos ou features).")
        else:
            print(f"\nCluster {cluster}: Poucos pontos para calcular correlação.")


def calculate_silhouette_for_k(df_processed_scaled, range_n_clusters):
    silhouette_scores = {}
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_processed_scaled) 
        silhouette_avg = silhouette_score(df_processed_scaled, cluster_labels)
        silhouette_scores[n_clusters] = silhouette_avg
        print(f"Silhouette Score para {n_clusters} clusters: {silhouette_avg:.4f}")
    return silhouette_scores


if __name__ == '__main__':
    dataset_path = 'data/processed/customers_dataset.csv'
    if os.path.exists(dataset_path):
        customers_df = pd.read_csv(dataset_path)
        print("Dataset já existente carregado.")
    else:
        customers_df = load_and_merge_data(
            raw_data_path='data/raw', 
            output_path=dataset_path,
            remove_raw=True
        )
        print("Dataset criado e salvo.")
    df_processed = preprocess_data(customers_df.copy())
    sample_size = int(0.2 * len(df_processed))
    df_processed_sampled = df_processed.sample(n=sample_size, random_state=42).copy()
    df_processed_scaled = df_processed_sampled.copy()
    numeric_cols_to_scale = df_processed_scaled.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_processed_scaled[numeric_cols_to_scale] = scaler.fit_transform(df_processed_scaled[numeric_cols_to_scale])

    print("----------------Dados pré-processados e prontos para a clusterização.----------------")
    print("\nColunas do dataframe:")
    print(df_processed.columns.tolist())
    show_descriptive_stats(customers_df)
    show_value_counts(customers_df)
    print("\n----------------Projeto iniciado e EDA concluída.----------------")
    print("----------------Dados escalonados.----------------")
    print("\nColunas do dataframe escalonado:")
    print(df_processed_scaled.columns.tolist())

    # Aplicação do método do cotovelo
    range_n_clusters = range(2, 11)

    print("\n----------------Aplicando KMeans na amostra:----------------")
    print("\n---Aplicando o Método do Cotovelo para avaliação de KMeans (na amostra)---")
    silhouette_scores_sampled = calculate_silhouette_for_k(df_processed_scaled.copy(), range_n_clusters)
    best_n_clusters_silhouette_sampled = max(silhouette_scores_sampled, key=silhouette_scores_sampled.get)
    print(f"\nO número ideal de clusters (segundo o Silhouette Score na amostra) é: {best_n_clusters_silhouette_sampled}")
    plt.figure(figsize=(10, 6))
    plt.plot(silhouette_scores_sampled.keys(), silhouette_scores_sampled.values(), marker='o')
    plt.title('Silhouette Score por Número de Clusters (KMeans na Amostra)')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(list(silhouette_scores_sampled.keys()))
    plt.grid(True)
    plt.show()
    plot_elbow_method(df_processed_scaled.copy(), range_n_clusters)

    #Melhor número de clusters, de acordo com o método do cotovelo
    best_n_clusters_elbow = best_n_clusters_silhouette_sampled
    df_kmeans = run_kmeans_clustering(df_processed_scaled.copy(), n_clusters=best_n_clusters_elbow)
    cluster_labels_kmeans = df_kmeans['Cluster']
    print(f"\n***Visualizando os clusters KMeans com PCA para {best_n_clusters_elbow} clusters:***")
    visualize_clusters_pca(df_processed_scaled, cluster_labels_kmeans, n_components=2, title='KMeans Clusters (PCA - 2 Componentes)')
    print(f"\n***Dataframe com rótulos de clusters KMeans para {best_n_clusters_elbow} clusters (Método do Cotovelo):***")
    print(df_kmeans.head())
    print("\n-----------Contagem de pontos por cluster (KMeans):-----------")
    print(df_kmeans['Cluster'].value_counts().sort_index())
    print(f"\n-----------Média das características por cluster (KMeans):-----------")
    df_kmeans_with_features = pd.concat([df_processed_scaled.reset_index(drop=True), cluster_labels_kmeans], axis=1)
    df_numeric_only = df_kmeans_with_features.drop(columns=df_kmeans_with_features.select_dtypes(include='object').columns)
    cluster_means_kmeans = df_numeric_only.groupby('Cluster').mean()
    print(cluster_means_kmeans)
    print("\n\n-----------Análise de correlação por clusters (KMeans): entre pares-----------")
    find_highly_correlated_features_per_cluster(df_kmeans_with_features, 'Cluster')
    numeric_cols_kmeans = df_kmeans_with_features.select_dtypes(include=np.number).columns.tolist()

    if 'Cluster' in numeric_cols_kmeans:
        numeric_cols_kmeans.remove('Cluster')
    features_to_analyze = numeric_cols_kmeans
    visualize_cluster_features_kmeans(df_kmeans_with_features, 'Cluster', features_to_analyze, title_prefix='KMeans')

    if len(np.unique(cluster_labels_kmeans)) > 1:
        silhouette_avg_kmeans = silhouette_score(df_processed_scaled, cluster_labels_kmeans)
        davies_bouldin_kmeans = davies_bouldin_score(df_processed_scaled, cluster_labels_kmeans)
        print("\n*****Métricas de avaliação (KMeans):*****")
        print_evaluation_metrics(best_n_clusters_elbow, silhouette_avg_kmeans, davies_bouldin_kmeans)
    else:
        print("\nNão foi possível calcular as métricas de avaliação para KMeans com apenas um cluster.")

    print("----------------Aplicando DBSCAN----------------")
    df_sample = df_processed_scaled.sample(n=len(df_processed_scaled), random_state=42) 
    print("\n----------------Avaliando parâmetros do DBSCAN----------------")
    plot_k_nearest_neighbors_eps(df_sample, k=5)
    eps_values = [0.2, 0.25, 0.3]
    min_samples_values = range(9, 19)
    dbscan_results_df = evaluate_dbscan_parameters(df_sample, eps_values, min_samples_values)
    print("\n******Resultados da avaliação dos parâmetros do DBSCAN******")
    print(dbscan_results_df)
    best_eps_sample = dbscan_results_df.iloc[0]['eps'] if not dbscan_results_df.empty else 3.1
    best_min_samples_sample = int(dbscan_results_df.iloc[0]['min_samples']) if not dbscan_results_df.empty else 9
    print(f"\nMelhores parâmetros DBSCAN estimados: eps={best_eps_sample}, min_samples={best_min_samples_sample}")
    df_dbscan_sample = run_dbscan_clustering(df_sample.copy(), eps=best_eps_sample, min_samples=best_min_samples_sample)
    cluster_labels_dbscan_sample = df_dbscan_sample['DBSCAN_Cluster']
    find_highly_correlated_features_per_cluster(df_dbscan_sample, 'DBSCAN_Cluster')
    numeric_cols_dbscan = df_dbscan_sample.select_dtypes(include=np.number).columns.tolist()
    
    if 'DBSCAN_Cluster' in numeric_cols_dbscan:
        numeric_cols_dbscan.remove('DBSCAN_Cluster')

    features_to_analyze_dbscan = numeric_cols_dbscan
    visualize_cluster_features_dbscan(df_dbscan_sample, 'DBSCAN_Cluster', features_to_analyze_dbscan, title_prefix=f"DBSCAN (eps={best_eps_sample}, ms={best_min_samples_sample}) ")
    print("\nContagem de pontos por cluster (DBSCAN):")
    print(cluster_labels_dbscan_sample.value_counts().sort_index())
    visualize_clusters_pca(df_sample, cluster_labels_dbscan_sample, n_components=2, title=f"DBSCAN Clusters (PCA - 2 Componentes) (eps={best_eps_sample}, min_samples={best_min_samples_sample})", filename_suffix=f"dbscan_eps{best_eps_sample}_ms{best_min_samples_sample}")
    df_dbscan_with_features_sample = pd.concat([df_sample.reset_index(drop=True), cluster_labels_dbscan_sample], axis=1)
    visualize_cluster_features_dbscan(df_dbscan_with_features_sample, 'DBSCAN_Cluster', features_to_analyze_dbscan, title_prefix=f"DBSCAN (eps={best_eps_sample}, ms={best_min_samples_sample}) ")
    core_samples_mask_dbscan = np.array(cluster_labels_dbscan_sample) != -1
    labels_dbscan_no_noise = np.array(cluster_labels_dbscan_sample)[core_samples_mask_dbscan]
    n_clusters_dbscan = len(np.unique(labels_dbscan_no_noise))
    
    if n_clusters_dbscan > 1:
        silhouette_avg_dbscan = silhouette_score(df_sample[core_samples_mask_dbscan], labels_dbscan_no_noise)
        davies_bouldin_dbscan = davies_bouldin_score(df_sample[core_samples_mask_dbscan], labels_dbscan_no_noise)
        print("\n****Métricas de avaliação DBSCAN (sem ruído):****")
        print_evaluation_metrics(n_clusters_dbscan, silhouette_avg_dbscan, davies_bouldin_dbscan)
    else:
        print("\nNão foi possível calcular as métricas de avaliação para DBSCAN com menos de dois clusters (sem ruído).")

    print("\n----------------Aplicando HDBSCAN----------------")
    df_sample_hdbscan = df_sample.copy()
    min_cluster_size_values = [5, 10, 15, 20]
    min_samples_values = [None, 5, 10]

    print("\n----------------Avaliando parâmetros (HDBSCAN)----------------")
    hdbscan_results_df = evaluate_hdbscan_parameters(df_sample, min_cluster_size_values, min_samples_values)
    print("\n*****Resultados da avaliação dos parâmetros HDBSCAN:*****")
    print(hdbscan_results_df)
    best_mcs_hdbscan_sample = int(hdbscan_results_df.iloc[0]['min_cluster_size']) if not hdbscan_results_df.empty else 10
    best_ms_hdbscan_sample_raw = hdbscan_results_df.iloc[0]['min_samples'] if not hdbscan_results_df.empty else None
    best_ms_hdbscan_sample = int(best_ms_hdbscan_sample_raw) if pd.notna(best_ms_hdbscan_sample_raw) else None
    print(f"\n----------------Melhores parâmetros (estimados (HDBSCAN): min_cluster_size={best_mcs_hdbscan_sample}, min_samples={best_ms_hdbscan_sample}----------------)")
    df_hdbscan_sample = run_hdbscan_clustering(df_sample.copy(), min_cluster_size=best_mcs_hdbscan_sample, min_samples=best_ms_hdbscan_sample)
    cluster_labels_hdbscan_sample = df_hdbscan_sample['HDBSCAN_Cluster']
    print("\nContagem de pontos por cluster (HDBSCAN):")
    print(cluster_labels_hdbscan_sample.value_counts().sort_index())
    print(f"\nVisualizando os clusters HDBSCAN com PCA:")
    visualize_clusters_pca(df_sample, cluster_labels_hdbscan_sample, n_components=2, title='HDBSCAN Clusters (PCA - 2 Componentes)', filename_suffix=f"hdbscan_sample_pca_2d_mcs{best_mcs_hdbscan_sample}_ms{best_ms_hdbscan_sample}")
    df_hdbscan_with_features_sample = pd.concat([df_sample.reset_index(drop=True), cluster_labels_hdbscan_sample], axis=1)
    numeric_cols_hdbscan_with_labels = df_hdbscan_with_features_sample.select_dtypes(include=np.number).columns.tolist()

    if 'HDBSCAN_Cluster' in numeric_cols_hdbscan_with_labels:
        numeric_cols_hdbscan_with_labels.remove('HDBSCAN_Cluster')
    features_to_analyze_hdbscan = numeric_cols_hdbscan_with_labels

    visualize_cluster_features_hdbscan(df_hdbscan_with_features_sample, 'HDBSCAN_Cluster', features_to_analyze_hdbscan, title_prefix='HDBSCAN')
    labels_hdbscan_no_noise = np.array(cluster_labels_hdbscan_sample)[np.array(cluster_labels_hdbscan_sample) != -1]
    n_clusters_hdbscan = len(np.unique(labels_hdbscan_no_noise))

    if n_clusters_hdbscan > 1:
        silhouette_avg_hdbscan = silhouette_score(df_sample[np.array(cluster_labels_hdbscan_sample) != -1], labels_hdbscan_no_noise)
        davies_bouldin_hdbscan = davies_bouldin_score(df_sample[np.array(cluster_labels_hdbscan_sample) != -1], labels_hdbscan_no_noise)
        print("\n*****Metricas de avaliação HDBSCAN (sem ruído):*****")
        print_evaluation_metrics(n_clusters_hdbscan, silhouette_avg_hdbscan, davies_bouldin_hdbscan)
    else:
        print("\nNão foi possível calcular métricas de avaliação HDBSCAN com menos de dois clusters (sem ruído).")

    print("\n--------------------------------Fluxo principal concluído.--------------------------------")
    print("\n----------------Carregamento, pré-processamento, clusterização (KMeans, DBSCAN, HDBSCAN) e avaliação OK----------------")
    print("\nMétricas de avaliação e visualização concluídas com sucesso.")