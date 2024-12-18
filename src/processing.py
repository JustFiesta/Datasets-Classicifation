import pandas as pd

def load_and_clean_data(csv_path):
    """
    Ładuje dane z pliku CSV, obsługuje brakujące wartości i zwraca przetworzoną ramkę danych.
    
    Args:
        csv_path (str): Ścieżka do pliku CSV z danymi.
    Returns:
        pd.DataFrame: Przetworzona ramka danych.
    """
    try:
        data = pd.read_csv(csv_path)

        # Łączenie kolumn 'Subject' i 'Message'
        data['Combined_Message'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')

        # Usuwanie niepotrzebnych kolumn
        data = data.drop(columns=['Subject', 'Message'])

        # Obsługa brakujących wartości
        data = data.dropna(subset=['Spam/Ham'])

        # Konwersja kolumny 'Spam/Ham' na etykiety binarne
        data['Spam'] = data['Spam/Ham'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

        # Usunięcie kolumny 'Spam/Ham'
        data = data.drop(columns=['Spam/Ham'])

        return data

    except Exception as e:
        print(f"Błąd podczas przetwarzania danych: {e}")
        return None
