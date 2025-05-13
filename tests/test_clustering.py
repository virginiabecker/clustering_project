import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.clustering import run_kmeans_clustering
import pytest


@pytest.fixture(scope="module")
def prepared_data():
    data_path = 'data/customers_dataset.csv'
    df = load_data(data_path)
    df_processed = preprocess_data(df.copy())
    return df, df_processed

def test_run_kmeans_clustering(prepared_data):
    df, df_processed = prepared_data
    num_clusters = 3

    df_numeric = df_processed.select_dtypes(include=['int64', 'float64'])
    df_clustered = run_kmeans_clustering(df_numeric.copy(), n_clusters=num_clusters)
    assert df_clustered['Cluster'].nunique() == num_clusters, f"Esperado {num_clusters} clusters, mas encontrado {df_clustered['Cluster'].nunique()}."
    assert 'Cluster' in df_clustered.columns, "A coluna 'Cluster' não foi encontrada."
    assert df_clustered['Cluster'].dtype in ['int32', 'int64'], "A coluna 'Cluster' não tem valores inteiros."
    assert df_clustered['Cluster'].nunique() <= num_clusters, f"Número de clusters ({df_clustered['Cluster'].nunique()}) maior que o solicitado ({num_clusters})."
    assert df_clustered.shape[0] == df_processed.shape[0], "O número de linhas com cluster deve ser igual ao original."
    assert df_clustered['Cluster'].isnull().sum() == 0, "Existem valores nulos na coluna 'Cluster'."  
    
    assert df_clustered.shape[0] == df_processed.shape[0], "O número de linhas com cluster deve ser igual ao original."
    assert df_clustered['Cluster'].isnull().sum() == 0, "Existem valores nulos na coluna 'Cluster'."

    print("Teste do KMeans passou com sucesso.")

