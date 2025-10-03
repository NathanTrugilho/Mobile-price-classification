import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(selected_features=None):
    df = pd.read_csv('dataset/mobile_price_classification_dataset.csv')
    
    target_column = 'price_range' #Coluna q vou prever

    if selected_features is not None:
        X = df[selected_features]
    else:
        X = df.drop(target_column, axis=1) # Se eu não definir nada, divide usando todos os parâmetros do dataset

    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=28,
        stratify=y, # garante que a divisão dos dados em conjuntos de treino e teste mantenha a mesma proporção de cada classe que existia no conjunto de dados original
    )

    return X_train, X_test, y_train, y_test
