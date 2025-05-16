import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler

def apply_label_encoding(df, categorical_cols):
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def preprocess_data(df):
    df = df.copy()
    cols_to_drop = [
        'review_comment_message',
        'review_comment_title',
        'shipping_limit_date',
        'order_purchase_timestamp',
        'customer_zip_code_prefix',
        'customer_city',
        'customer_unique_id',
        'freight_value',
        'order_delivered_carrier_date',
        'order_id',
        'order_item_id',
        'order_status',
        'payment_sequential',
        'payment_installments',
        'payment_value',
        'price',
        'product_id',
        'product_photos_qty',
        'product_description_lenght',
        'product_weight_g',
        'product_name_lenght',
        'product_category_name',
        'review_score',
        'seller_zip_code_prefix',
        'seller_city',
        'seller_id',
        'seller_state',
        'customer_state',
        'payment_type',
        'review_id',
        'review_creation_date',
        'review_answer_timestamp',
        'order_estimated_delivery_date'
    ]

    #categorical_cols = ['payment_type']
    #df = apply_label_encoding(df, categorical_cols)
    #df['vol_by_product'] = df['product_height_cm'] * df['product_width_cm'] * df['product_length_cm']
    df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['days_to_delivery'] = (df['order_delivered_customer_date'] - df['order_approved_at']).dt.days
    df.drop(columns=['order_approved_at', 'order_delivered_customer_date'], inplace=True)
    df = df.dropna(subset=['lat_customer', 'lng_customer', 'lat_seller', 'lng_seller'])
    df['distance_km'] = df.apply(lambda row: geodesic(
       (row['lat_seller'], row['lng_seller']),
       (row['lat_customer'], row['lng_customer'])
        ).km, axis=1
        )
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    df.drop(
        columns=[
            'product_height_cm', 'product_width_cm', 'product_length_cm',
            'lat_customer', 'lng_customer', 'lat_seller', 'lng_seller', 'customer_id',], 
            inplace=True
        )
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'uint8']).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    print(df.columns)
    return df

if __name__ == '__main__':
    df = pd.read_csv("data/processed/customers_dataset.csv", encoding='utf-8', on_bad_lines='skip')
    df_proc = preprocess_data(df)
    df_proc.to_csv('data/processed/output.csv', index=False)

    
