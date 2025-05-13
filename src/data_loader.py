import os
import pandas as pd

def load_and_merge_data(raw_data_path='data/raw', output_path='data/processed/customers_dataset.csv', remove_raw=False):
    print("Lendo arquivo...")

    customers = pd.read_csv(os.path.join(raw_data_path, 'olist_customers_dataset.csv'))
    orders = pd.read_csv(os.path.join(raw_data_path, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(raw_data_path, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(raw_data_path, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(raw_data_path, 'olist_sellers_dataset.csv'))
    reviews = pd.read_csv(os.path.join(raw_data_path, 'olist_order_reviews_dataset.csv'))
    
    #print("Colunas de 'orders':", orders.columns)
    #print("Colunas de 'customers':", customers.columns)

    print("Fazendo merge...")
    orders['order_id'] = orders['order_id'].astype(str)
    reviews['order_id'] = reviews['order_id'].astype(str)

    df = orders.merge(customers, on='customer_id', how='left') \
               .merge(order_items, on='order_id', how='left') \
               .merge(products, on='product_id', how='left') \
               .merge(sellers, on='seller_id', how='left') \
               .merge(reviews, on='order_id', how='left') \

            
    print(f"Salvando dataset final em: {output_path}")
    df.to_csv(output_path, index=False)

    if remove_raw:
        for file in os.listdir(raw_data_path):
            os.remove(os.path.join(raw_data_path, file))
        print("Arquivos brutos removidos.")


    return df

def load_data(datapath):
    if not os.path.exists(datapath):
        print(f"{datapath} não encontrado. Gerando arquivo...")
        load_and_merge_data()
    return pd.read_csv(datapath)

def data_overview(df):
    print("\n***Primeiras linhas:***")
    print(df.head())

    print("\n***Descrição estatística:***")
    print(df.describe(include='all'))

    print("\nInformações do dataframe:")
    print(df.info())

    print("\nColunas do dataframe:")
    for col in df.columns:
        print(col)

    print(f"\nQuantidade de pedidos únicos (order_id): {df['order_id'].nunique()}")

datapath = 'data/processed/customers_dataset.csv'

if __name__ == '__main__':
    df = load_data(datapath)
    #data_overview(df)
    print(df.columns)