from src.data_loader import load_data 
from src.preprocessing import preprocess_data

def show_descriptive_stats(df):
    numeric_df = df.select_dtypes(include=['number'])

    if not numeric_df.empty:
        print("\n****Estatísticas descritivas das variáveis numéricas:****")
        print(numeric_df.describe())

    else:
        print("\nNão há colunas numéricas.")



def show_value_counts(df):
    categorical_df = df.select_dtypes(include=['object', 'category'])

    if not categorical_df.empty:
        print("\n***Contagem de valores categóricos:***")
        for col in categorical_df.columns:
            print(f"\nColuna: {col}")
            print(df[col].value_counts())

    else:
        print("\nNão há colunas categóricas.")

if __name__ == '__main__':
    data_path = 'data/eda/customers_dataset.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    show_descriptive_stats(df)
    show_value_counts(df)
