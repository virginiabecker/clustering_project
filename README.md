# Clusterização de Clientes com DBSCAN: Explorando Volume, Pagamento e Proximidade

Este projeto explora a aplicação de técnicas de clusterização para segmentar clientes de uma plataforma de e-commerce. O objetivo principal é identificar grupos com perfis semelhantes quanto ao valor total pago, à quantidade de parcelas e ao número de itens comprados. Através da análise comparativa de diferentes algoritmos de clusterização (KMeans, DBSCAN e HDBSCAN), o estudo demonstra a superioridade do DBSCAN na identificação de segmentos mais coesos e bem definidos. 

## Principais etapas

### 1. Análise Exploratória de Dados (EDA)
Nesta etapa, foram calculadas as estatísticas descritivas das variáveis numéricas com a função `show_descriptive_stats`. Esta etapa permitiu obter uma visão geral da distribuição e da escala dos dados numéricos.
Para as variáveis categóricas, foi realizada a contagem da frequência de cada valor único utilizando a função `show_value_counts`. Isso ajudou a entender a distribuição das categorias e identificar possíveis desbalanceamentos.

### 2. Pré-processamento
Diversas colunas consideradas não relevantes para a análise foram removidas do dataset para diminuir a dimensionalidade e o ruído. Identificadores, informações textuais, datas específicas e detalhes com especificidades de produtos.
As colunas `order_approved_at` e `order_delivered_customer_date` foram convertidas para o tipo datetime, de modo que permitisse cálculos temporais.
Criação da feature numérica `days_to_dalivery` representando tempo de entrega em dias (a partir das features `order_approved_at` e `order_delivered_customer_date` - posteriormente removidas).
O NaNs (valores ausentes) foram preenchidos com 0.
Por fim, as variáveis numéricas foram escalonadas por meio do StandardScaler, que ajuda a padronizar a escala das features.

### 3. Clusterização
Nesta etapa, foram implementados três algoritmos de clusterização utilizando a biblioteca scikit-learn.
São eles: 
**KMeans:** a função `run_kmeans_clustering` foi utilizada. `n_clusters` foi definido como 8, conforme determinado pela análise do método do cotovelo.
**DBSCAN:** a função `run_dbscan_clustering` foi utilizada. Os parâmetros `eps` e `min_samples` foram definidos com base na avaliação de parâmetros em uma amostra do dataset (`eps=3.1`, `min_samples=9`).
**HDBSCAN:** a função `run_hdbscan_clustering` foi utilizada. Os parâmetros `min_cluster_size` e `min_samples` foram definidos com base na avaliação de parâmetros em uma amostra do dataset (`min_cluster_size=10`, `min_samples=None`).
Cada algoritmo foi aplicado ao dataset pré-processado para identificar grupos de clientes com padrões semelhantes nas features de valor e desempenho de entrega.
A qualidade dos clusters gerados por cada algoritmo foi avaliada utilizando métricas como o silhouette-score e o índice de davis-bouldin. Seus resultados foram comparados para determinar o algoritmo mais adequado para a segmentação dos dados.
Também foram aplicadas técnicas de redução de dimensionalidade (PCA) para visualização dos clusters em 2D, auxiliando na interpretação dos resultados. As características foram analisadas através da visualização de distribuição das features em cada grupo.


## Análise dos Resultados da Clusterização

**Métricas de avaliação encontradas para cada algoritmo de clusterização**

KMeans - Para 2 clusters:
* Silhouette score: 0.4462
* Índice de davies-bouldin: 1.2066

DBSCAN - Para 21 clusters:
* Silhouette score: 0.4994
* Índice de davies-bouldin: 0.7478

HDBSCAN - Para 1614 clusters:
* Silhouette score: 0.5783
* Índice de davies-bouldin: 0.5090


### Análise com KMeans:
A partir do método do cotovelo, identificou-se 2 como o melhor número de clusters para se analisar. 

![Melhor n_clusters de acordo com o método do cotovelo](elbow_method.png)

A partir desta busca por diferentes correlações entre days_to_delivery e distance_km foram obtidos 0,3962 para o cluster 0 e 0.4047 para o cluster 1.
Os clusters apresentaram correlações relevantes entre as features, porém, ao aplicar o silhouette score e o índice de davis-bouldin evidenciou-se baixa coesão dos valores, especialmente no quesito separabilidade (com enfoque para o valor do índice de davis-bouldin: > 1) com o que os clusters apresentaram. Daí, partimos para a próxima análise, utilizando o parâmetro DBSCAN.

### Análise com DBSCAN:
 O DBSCAN apresentou-se como um bom caminho para a nossa análise, revelando clusters robustos, com boa densidade e pouco ruído. 

 ![Melhor n_clusters de acordo com o método do cotovelo](knn_eps_estimation.png)

 Apesar da sobreposição visual entre pontos nos gráficos, o DBSCAN na visualização com PCA em 2d reforçou essa estrutura, validando a segmentação. Contudo, a análise das distribuições revelou a presença significativa de outliers no Cluster 2 (cluster com maior número de pontos). Esse aspecto é relevante e deve ser considerado na interpretação dos resultados.
De maneira geral, os prazos de entrega (days_to_delivery) apresentaram medianas consistentes entre os clusters, com poucas variações. 
 Já a distância percorrida mostrou variações mais expressivas, sendo um diferencial importante entre os perfis de cada grupo.

 ![Distribuição das distâncias por região](cluster_visualization_per_feature_distance_km_dbscan.png)
 ![Distribuição dos dias para entrega do pedido](cluster_visualization_per_feature_days_to_delivery_dbscan.png)

#### Clusters com maior proximidade
* Cluster 20: concentrou clientes localizados em áreas significativamente mais próximas.
* Clusters 9, 13 e 18: também mostraram distâncias reduzidas, com prazos medianos de entrega entre os menores observados.
* Estes grupos não apresentaram outliers relevantes nem em prazo, nem em distância.

#### Clusters estáveis em distância e prazo
* Clusters 0 a 6: apresentaram comportamentos muito semelhantes em termos de distância, apenas com algumas variações discretas entre si.
* Cluster 2: muitos outliers em distância e prazo de entrega, indicando maior variabilidade num geral.
* Clusters 3, 4 e 6: também com outliers, mas em menor intensidade.
#### Clusters mais diversos:
* Clusters 7, 14, 18, 19 e 20: não apresentaram outliers de distância, sugerindo distribuição mais compacta.
* Cluster 11: embora discreto quanto à distância, apresentou o maior prazo de entrega registrado, evidenciado por um outlier acima de 17.5 dias.

#### Considerações
Com base nas observações levantadas, o cluster 2 merece atenção especial, dada sua representatividade em número de pontos e por sua alta variabilidade nos dois eixos: o que pode indicar desafios logísticos ou abrangência geográfica mais ampla.
Já os clusters com menor distância e prazos consistentes podem representar regiões urbanas bem atendidas ou clientes localizados próximos aos centros de distribuição.

### Análise com HDBSCAN:
Tanto em parâmetros de avaliação quanto nas análises gráficas, os dados mostram que para esta análise o HDBSCAN mostrou um desempenho inferior em comparação com o DBSCAN. 
O HDBSCAN obteve um número significativamente maior de clusters (1614), tornando-o um algoritmo inviável para esta análise, apesar de apresentar melhores valores de silhouette score e índice de davis-bouldin.

### DBSCAN como métrica principal
O índice de davies-bouldin reforça ainda mais o que obtivemos com o silhouette score do DBSCAN, dado que é uma métrica que se deseja atingir valores menores. Apesar de o HDBSCAN ter apresentado melhores níveis de ambos, pelo número de clusters avaliado não foi possível aplicá-lo neste projeto. Como o HDBSCAN capta as densidades, e o dataset que utilizamos possui muitas granulações, sua captura através do algoritmo forma muitos clusters, não sendo possível construir uma análise clara e objetiva para o que o projeto se propõe.
Ao fim, a análise quantitativa através do silhouette score e do índice de davis-bouldin apontou consistentemente para uma qualidade de clusterização fraca por parte do KMeans e HDBSCAN em comparação com os resultados notavelmente superiores obtidos pelo DBSCAN em nossa amostra. O DBSCAN, com sua capacidade de identificar clusters de densidade variável e lidar com ruído de forma eficaz, demonstrou ser a abordagem mais adequada para segmentar os dados deste projeto. 


## Organização do Projeto

```bash
CLUSTER_PROJECT/
├── data/                     
│   ├── processed/            
│   │   └── customers_dataset.csv
│   └── raw/                 
├── src/                     
│   ├── data_loader.py        
│   ├── preprocessing.py      
│   ├── exploratory_analysis.py 
│   ├── clustering.py         
│   ├── model_evaluation.py   
│   └── visualization.py      
├── reports/                 
│   ├── clusters/             
│   └── eda/                  
├── tests/                    
│   ├── test_data_loader.py  
│   ├── test_preprocessing.py 
│   ├── test_clustering.py    
│   └── test_evaluation.py   
├── venv/                       
├── main.py                   
├── pytest.ini                
├── README.md                 
└── requirements.txt          
```


## Conjunto de Dados
O dataset utilizado é o dataset público da Olist: Brazilian E-Commerce, fundamentado em 9 planilhas iniciais, sendo destas 8 selecionadas para a análise: olist_customers_dataset, olist_order_items_dataset, olist_order_reviews_dataset, olist_orders_dataset, olist_products_dataset, olist_payments_dataset, olist_geolocation_dataset e olist_sellers_dataset.

Após tratamento dos dados, estas planilhas foram unidas e formaram o dataset inicial `customers_dataset`.
Nele, estão contidas as informações sobre transações e avaliações. As principais variáveis no dataset (para a clusterização) são:

**Variáveis originais**
order_approved_at: data e hora em que o pagamento foi aprovado.
order_delivered_customer_date: data e hora em que o pedido foi entregue ao cliente.
geolocation_lat: latitude para referência.
geolocation_lng: longitude para referência.
geolocation_zip_code_prefix: prefixo para referência.
seller_zip_code_prefix: prefixo da região do centro de distribuição.
customer_zip_code_prefix: prefixo da região do cliente.

**Variável derivada das originais para análise**
days_to_delivery: número de dias entre a data da compra e a data da entrega.
distance_km: distância entre o cliente e o centro de distribuição.

## Como Usar:

### 1. **Clonar o repositório:**
```bash
git clone
cd cluster_project
```

### 2. **Criar e ativar o ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

### 3. **Instalar as dependências:**
```bash
pip install -r requirements.txt
```

### 4. **Executar o projeto:**
```bash
python main.py
```

## Bibliotecas Utilizadas
* pandas
* numpy
* scikit-learn
* matplotlib
* geopy
* seaborn
* hdbscan

## Autora

**Virginia Becker**

Economista em transição para a área de dados, apaixonada por resolver problemas com análise, machine learning e boas histórias com dados.

#### [Lindedin](https://www.linkedin.com/in/virginia-becker/) • [Medium](https://medium.com/@virginia.becker)