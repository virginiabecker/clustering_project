import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

def visualize_clusters_pca(df, labels, n_components=2, title='Visualização de clusters com PCA', filename_suffix=''):
    numeric_df = df.select_dtypes(include=['int64', 'float64', 'uint8'])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Cluster'] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title(title)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.savefig(f'reports/clusters/cluster_visualization_pca_{n_components}d_{filename_suffix}.png')
    plt.show()

    if n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['Cluster'], cmap='viridis')
        ax.set_xlabel('Componente principal 1')
        ax.set_ylabel('Componente principal 2')
        ax.set_zlabel('Componente principal 3')
        ax.set_title('***Visualização de clusters com uso de PCA (3 Componentes):***')
        fig.colorbar(scatter, ax=ax, label='Cluster')
        plt.savefig('reports/clusters/cluster_visualization_pca_3d.png')
        plt.show()

def visualize_cluster_features_kmeans(df, cluster_column, features, title_prefix=''):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_column, y=feature, data=df)
        plt.title(f'{title_prefix} - Distribuição de {feature} por Cluster')
        plt.savefig(f'reports/clusters/cluster_visualization_per_feature_{feature}_kmeans.png')
        plt.show()

def visualize_cluster_features_dbscan(df, cluster_column, features, title_prefix=''):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_column, y=feature, data=df)
        plt.title(f'{title_prefix} - Distribuição de {feature} por Cluster')
        plt.savefig(f'reports/clusters/cluster_visualization_per_feature_{feature}_dbscan.png')
        plt.show()

def visualize_cluster_features_hdbscan(df, cluster_column, features, title_prefix=''):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_column, y=feature, data=df)
        plt.title(f'{title_prefix} - Distribuição de {feature} por Cluster')
        plt.savefig(f'reports/clusters/cluster_visualization_per_feature_{feature}_hdbscan.png')
        plt.show()