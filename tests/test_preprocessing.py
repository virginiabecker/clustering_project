import pandas as pd
import pytest
from src.data_loader import load_data
from src.preprocessing import preprocess_data

def test_preprocess_data():
    data_path = 'data/customers_dataset.csv'
    customers_df = load_data(data_path)
    df = customers_df.copy()
    df_processed = preprocess_data(df)

    cols_to_remove = [
        'shipping_limit_date',
        'order_purchase_timestamp',
        'customer_zip_code_prefix',
        'customer_city',
        'review_comment_message',
        'review_comment_title',
        'review_answer_timestamp',
        'seller_zip_code_prefix',
        'seller_city',
        'review_id',
        'review_creation_date'
    ]

    for col in cols_to_remove:
        assert col not in df_processed.columns, f"{col} deve ser removido."
    categorical_cols = ['order_status', 'payment_type', 'customer_state']
    for col in categorical_cols:
        assert f'{col}_' in df_processed.columns, f"Falha no one-hot encoding de {col}."

    numeric_cols = df_processed.select_dtypes(include=['number'].columns.tolist())
    for col in numeric_cols:
        assert df_processed[col].mean() < 1e-6, f"Falha no escalonamento de {col} (média)."
        assert abs(df_processed[col].std()-1) < 1e-6, f"Falha no escalonamento de {col} (desvio padrão)."
    assert df_processed.shape[0] == customers_df.shape[0], "Número de linhas deve permanecer o mesmo."

    empty_df = pd.DataFrame()
    empty_df_processed = preprocess_data(empty_df.copy())
    assert empty_df_processed.empty, "Dados processados devem retornar um dataframe vazio, se entrada for vazia."

    df_no_order_id = customers_df.copy().drop(columns=['order_id'], errors='ignore')
    df_no_order_id_processed = preprocess_data(df_no_order_id.copy())
    assert 'order_id' not in df_no_numeric_processed.columns, "Função preprocess_data deve funcionar mesmo sem order_id."

    df_no_numeric = df.select_dtypes(exclude=['number'])
    df_no_numeric_processed = preprocess_data(df_no_numeric.copy())
    assert df_no_numeric_processed.select_dtypes(include=['number']).empty, "Função preprocess_data deve funcionar mesmo sem colunas numéricas."

    original_categorical_cols = ['order_status', 'payment_type', 'customer_state']
    num_original_categories = sum(customers_df[col].nuniqueuw() for col in original_categorical_cols)
    expected_encoded_cols = num_original_categories + df.shape[1] - len(original_categorical_cols)
    assert df_processed.shape[1] == expected_encoded_cols, f"Numero de colunas após encoding ({df_processed.shape[1]}) não corresponde ao esperado ({expected_encoded_cols})."

    for col in df_processed.columns:
        if any(col.startswith(c) for c in original_categorical_cols):
            assert df_processed[col].isin([0, 1]).all(), f"Coluna {col} do one-hot encoding contém valores diferentes de 0 ou 1."

pytest.main()