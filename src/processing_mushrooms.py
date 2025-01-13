import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

def load_and_clean_mushrooms_data():
    """
    Ładuje dane repozytorium UCI i obsługuje brakujące wartości i zwraca przetworzoną ramkę danych.

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Pobranie datasetu
    dataset = fetch_ucirepo(id=73)

    # Konwersja na dataframe
    X = pd.DataFrame(dataset.data.features)
    y = pd.DataFrame(dataset.data.targets)

    # Połączenie danych
    df = pd.concat([X, y], axis=1)

    # Obsługa pustych wartości
    df.ffill(inplace=True)

    # Enkodowanie wartości do kolumn
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    return df
