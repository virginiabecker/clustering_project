import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def apply_label_encoding(df, categorical_cols):
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def preprocess_data(df):
    cols_to_drop = [
        'review_comment_message',
        'review_comment_title',
        'shipping_limit_date',
        'order_purchase_timestamp',
        'customer_zip_code_prefix',
        'customer_city',
        'customer_id',
        'customer_unique_id',
        'order_delivered_carrier_date',
        'order_estimated_delivery_date',
        'order_item_id',
        'order_status',
        'order_id',
        'product_photos_qty',
        'product_description_lenght',
        'product_weight_g',
        'product_lenght_cm',
        'product_height_cm',
        'product_name_lenght',
        'product_width_cm',
        'seller_zip_code_prefix',
        'seller_city',
        'review_id',
        'review_creation_date',
        'review_answer_timestamp'
    ]

    df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['days_to_delivery'] = (df['order_delivered_customer_date'] - df['order_approved_at']).dt.days
    df.drop(columns=['order_approved_at', 'order_delivered_customer_date'], inplace=True)
    df['seller_id_numeric'] = pd.factorize(df['seller_id'])[0] + 1
    df.drop(columns=['seller_id'], inplace=True)
    avg_review_score_per_seller = df.groupby('seller_id_numeric')['review_score'].mean().reset_index()
    avg_review_score_per_seller.rename(columns={'review_score': 'avg_review_score_per_seller'}, inplace=True)
    avg_review_score_per_seller.to_csv("data/processed/avg_review_score_per_seller.csv", index=False)
    df = df.merge(avg_review_score_per_seller, on='seller_id_numeric', how='left')
    df.rename(columns={'review_score': 'review_score_individual'}, inplace=True)

    # Criação de novas features:
    df['freight_ratio'] = df['freight_value'] / df['price']
    df['freight_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['delivery_was_fast'] = (df['days_to_delivery'] < 5).astype(int)
    df['review_score_normalized'] = df['review_score_individual'] / df['avg_review_score_per_seller']
    df['review_score_normalized'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['loyal_customer_state'] = (df['customer_state'] == df['seller_state']).astype(int)
    top_categories = df['product_category_name'].value_counts().nlargest(5).index
    df['top_product'] = df['product_category_name'].isin(top_categories).astype(int)
    category_counts = df['product_category_name'].value_counts()
    rare_categories = category_counts[category_counts < 1-00].index
    df['rare_category'] = df['product_category_name'].isin(rare_categories).astype(int)
    sales_by_seller = df['seller_id_numeric'].value_counts()
    df['sales_vol_by_seller'] = df['seller_id_numeric'].map(sales_by_seller)
    df['avg_ticket_per_seller'] = df.groupby('seller_id_numeric')['price'].transform('mean')
    df['score_delivery'] = df['days_to_delivery'] + df['freight_ratio']

    categorical_cols = ['product_category_name', 'customer_state', 'seller_state', 'product_id']
    df = apply_label_encoding(df, categorical_cols)
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint8']).columns.tolist()
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(df.dtypes)

    return df

if __name__ == '__main__':
    df = pd.read_csv("data/processed/customers_dataset.csv", encoding='utf-8', on_bad_lines='skip')
    df_proc = preprocess_data(df)
    df_proc.to_csv('data/processed/output.csv', index=False)
    print(df_proc['price'])
