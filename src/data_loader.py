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
    geolocation = pd.read_csv(os.path.join(raw_data_path, 'olist_geolocation_dataset.csv'))
    payments =pd.read_csv(os.path.join(raw_data_path, 'olist_order_payments_dataset.csv'))
    customers = customers.rename(columns={'customer_zip_code_prefix': 'geolocation_zip_code_prefix'})
    sellers = sellers.rename(columns={'seller_zip_code_prefix': 'geolocation_zip_code_prefix'})


    customers = customers.rename(columns={'customer_zip_code_prefix': 'geolocation_zip_code_prefix'})
    sellers = sellers.rename(columns={'seller_zip_code_prefix': 'geolocation_zip_code_prefix'})

    geo = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()

    customers = customers.merge(geo, on='geolocation_zip_code_prefix', how='left')
    sellers = sellers.merge(geo, on='geolocation_zip_code_prefix', how='left')

    customers = customers.rename(columns={'geolocation_lat': 'lat_customer', 'geolocation_lng': 'lng_customer'})
    sellers = sellers.rename(columns={'geolocation_lat': 'lat_seller', 'geolocation_lng': 'lng_seller'})

    print("Fazendo merge...")
    orders['order_id'] = orders['order_id'].astype(str)
    reviews['order_id'] = reviews['order_id'].astype(str)

    df = orders.merge(customers, on='customer_id', how='left') \
               .merge(order_items, on='order_id', how='left') \
               .merge(products, on='product_id', how='left') \
               .merge(sellers, on='seller_id', how='left') \
               .merge(reviews, on='order_id', how='left') \
               .merge(payments, on= 'order_id')

    del geolocation
    columns_to_drop = ['geolocation_zip_code_prefix_x', 'geolocation_lat_x', 'geolocation_lng_x',
                       'geolocation_zip_code_prefix_y', 'geolocation_lat_y', 'geolocation_lng_y', 
                       'geolocation_zip_code_prefix','geolocation_lat','geolocation_lng',
                       'geolocation_city','geolocation_state']
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])    
    customers = customers.rename(columns={'geolocation_lat': 'lat_customer', 'geolocation_lng': 'lng_customer'})
    sellers = sellers.rename(columns={'geolocation_lat': 'lat_seller', 'geolocation_lng': 'lng_seller'})

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
    raw_path = 'data/raw'
    df = load_and_merge_data(raw_data_path=raw_path, output_path=datapath)
    print(df.columns)
    