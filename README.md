# Segmentação por Cluster: Uma Análise de Valor e Desempenho de Entrega com DBSCAN
Este projeto explora a aplicação de técnicas de clusterização para segmentar clientes de uma plataforma e-commerce. Tem-se como objetivo principal identificar grupos com perfis semelhantes em relação ao valor das compras realizadas (preço, valor do frete e proporção do frete) e ao desempenho da entrega dos pedidos (score de entrega). Por meio da análise comparativa de diferentes algoritmos de clusterização (KMeans, DBSCAN e HDBSCAN), este estudo demonstra a eficácia do DBSCAN na identificação de segmentos coesos e bem definidos. 

## Principais etapas

### 1. Análise Exploratória de Dados (EDA)
Nesta etapa, foram calculadas as estatísticas descritivas das variáveis numéricas com a função `show_descriptive_stats`. Esta etapa permitiu obter uma visão geral da distribuição e da escala dos dados numéricos.
Para as variáveis categóricas, foi realizada a contagem da frequência de cada valor único utilizando a função `show_value_counts`. Isso ajudou a entender a distribuição das categorias e identificar possíveis desbalanceamentos.

### 2. Pré-processamento
Diversas colunas consideradas não relevantes para a análise foram removidas do dataset para diminuir a dimensionalidade e o ruído. Identificadores, informações textuais, datas específicas e detalhes com especificidades de produtos.
As colunas `order_approved_at` e `order_delivered_customer_date` foram convertidas para o tipo datetime, de modo que permitisse cálculos temporais.
Criação da feature numérica `days_to_dalivery` representando tempo de entrega em dias (a partir das features `order_approved_at` e `order_delivered_customer_date` - posteriormente removidas).
Conversão da feature `seller_id` para `seller_id_numeric` com a técnica factorize, com a coluna original removida.
Criação da feature `avg_review_score_per_seller` para avaliar a reputação do vendedor, salva em uma nova planilha para referência futura.
A coluna `review_score` foi renomeada para `review_score_individual` para maior clareza.
Criação da feature `freight_ratio`, que representa a proporção entre o valor do frete e o preço pago pelo item, ou seja, uma medida relativa do custo de envio. 
Criação da feature `delivery_was_fast` (binária), que indica se a entrega ocorreu em menos de 5 dias.
Criação da feature `review_score_normalized`, que representa a nota individual normalizada pela média da nota do vendedor.
Features binárias foram criadas para indicar se o cliente e o vendedor estão no mesmo estado (loyal_customer_state), se o produto pertence a uma das 5 categorias mais vendidas (`top_product`) e se o produto pertence a uma categoria com baixa frequência de vendas('rare_category').
Foram criadas features indicando o volume de vendas por vendedor (`sales_vol_by_seller`) e o preço médio dos itens vendidos por vendedor (`avg_ticket_per_seller`).
Criação de `score_delivery`, que indica o desempenho da entrega, por meio da soma de dias para entrega e proporção do frete.
Colunas categóricas restantes (`product_category_name`, `customer_state`, `seller_state`) foram convertidas para representações numéricas com o label encoding.
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
Também foram aplicadas técnicas de redução de dimensionalidade (PCA) para visualização dos clusters em 2D e 3D, auxiliando na interpretação dos resultados. As características foram analisadas através da visualização de distribuição das features em cada grupo.


## Análise dos Resultados da Clusterização

###Métricas de avaliação encontradas para cada algoritmo de clusterização

KMeans - Para 8 clusters:
* Silhouette-score: 0.1293
* Índice de davies-bouldin: 1.7919

DBSCAN (amostra, excluindo ruído) - Para 2 clusters:
* Silhouette-score: 0.6146
* Índice de davies-bouldin: 0.5295

HDBSCAN (amostra, excluindo ruído) - Para 13 clusters:
* Silhouette-score: 0.1138
* Índice de davies-bouldin: 1.4026

### Análise com KMeans:
A partir do método do cotovelo, identificou-se 8 como o melhor número de clusters para se analisar. A partir desta busca por diferentes correlações entre pares, identificaram-se alguns pontos fortes (entre avg_ticket_per_seller e price de 0.872619 e entre score_delivery e days_to_delivery de 0.999885).
Os clusters apresentaram correlações relevantes entre as features, porém, ao aplicar o silhouette score evidenciou-se baixa coesão dos valores com o que os clusters apresentaram. Daí, partimos para a próxima análise, utilizando o parâmetro DBSCAN.

### Análise com DBSCAN:
Clusters robustos, com pouquíssimo ruído e boa densidade, apresentou-se como um ótimo caminho para a nossa análise, podendo verificar estas características com o uso de PCA em 2d e 3d, que fortalece as evidências. No entanto, ao analisarmos os níveis de distribuição das features, notou-se muitos outliers no Cluster 0, e isso não pode ser ignorado no estudo. A partir daí, podemos tirar algumas conclusões sobre os dados analisados:
•	Cluster 0: apresentou valores atípicos (outliers) para preços, valores de frete e score de entrega. No entanto, o padrão apresentado por este segmento é o de preços mais baixos, valores menores de frete e baixos scores de entrega. Além disso, há outliers que evidenciam frete caro em relação ao preço.
•	Cluster 1: evidenciou valores preços mais altos, valores maiores de frete e maiores scores de entrega. 
•	Cluster -1 (ruído): Pouca representação de ruído, mostrou-se nas 4 features, porém captando baixos valores e sem grandes variações.

### Análise com HDBSCAN:
Tanto em parâmetros de avaliação quanto nas análises gráficas, os dados mostram que para esta análise o HDBSCAN mostrou um desempenho inferior em comparação com o DBSCAN. 
O HDBSCAN mostrou um número significativamente maior de clusters (13), o que corrobora para um nível de silhouette score muito menor (menor coesão) e maior índice de davis-bouldin (maior similaridade entre os clusters).

### DBSCAN como métrica principal
O silhouette score de 0.1138 trazem uma coesão fraca entre os clusters e indica possível sobreposição entre eles (próximo de 0), contrastando fortemente com o silhouette score apresentado pelo DBSCAN de 0.6146, com uma estrutura de clusters bem mais definida e robusta. Já o índice de davies-bouldin reforça ainda mais o que obtivemos com o silhouette score, dado que é uma métrica que se deseja atingir valores menores: HDBSCAN obteve índice de davies-bouldin de 1.4026, enquanto DBSCAN foi de 0.5295, demonstrando maior coesão interna frente ao HDBSCAN. 
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
O dataset utilizado é o dataset público da Olist: Brazilian E-Commerce, fundamentado em 9 planilhas iniciais, sendo destas apenas 6 selecionadas para a análise: olist_customers_dataset, olist_order_items_dataset, olist_order_reviews_dataset, olist_orders_dataset, olist_products_dataset e olist_sellers_dataset.

Após tratamento dos dados, estas planilhas foram unidas e formaram o dataset inicial `customers_dataset`.
Nele, estão contidas as informações sobre transações e avaliações. As principais variáveis no dataset são:
**Variáveis originais**
customer_id: identificador de cliente
product_id: identificador do produto
seller_id: identificador do vendedor
price: preço do item
freight_value: valor do frete
order_approved_at: data e hora em que o pagamento foi aprovado.
order_delivered_customer_date: data e hora em que o pedido foi entregue ao cliente.
review_score: nota de avaliação dada pelo cliente (1 a 5)
product_category_name: nome da categoria do produto - em português
customer_state: Estado de cada cliente
customer_seller: Estado de cada vendedor
order_id: identificador de cada pedido
order_item_id: identificador do item dentro do pedido

**Variáveis derivadas das originais para análise**
days_to_delivery: número de dias entre a data da compra e a data da entrega
score_delivery: avaliação da entrega
freight_ratio: razão entre valor do frete e preço do item
review_score_normalized: versão normalizada da nota de avaliação
review_score_individual: avaliação por item do pedido
avg_review_score_per_seller: média da nota de avaliação por vendedor
avg_ticket_per_seller: valor médio do ticket por vendedor



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
* plotly
* geopy
* seaborn
* hdbscan

## Autora

**Virginia Becker**

Economista em transição para a área de dados, apaixonada por resolver problemas com análise, machine learning e boas histórias com dados.

#### [Lindedin](https://www.linkedin.com/in/virginia-becker/) • [Medium](https://medium.com/@virginia.becker)