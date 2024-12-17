"""
Preprocessing i zwrócenie danych z csv
"""
import pandas as pd

def load_and_clean_data(csv_path):
    """
    Ładuje dane z pliku CSV, obsługuje brakujące wartości i zwraca przetworzoną ramkę danych.
    
    :param csv_path: Ścieżka do pliku CSV z danymi.
    :return: Przetworzona ramka danych.
    """
    try:
        data = pd.read_csv(csv_path)

        # Sprawdzenie struktury danych
        print("Podgląd danych:")
        print(data.head())
        print(".")
        print(".")
        print(".")
        print(data.tail())

        # Obsługa brakujących wartości
        # Łączenie kolumn 'Subject' i 'Message' (zakładając, że 'Message' może być pusta)
        data['Combined_Message'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')

        # Usunięcie kolumn, które nie będą potrzebne po połączeniu
        data = data.drop(columns=['Subject', 'Message'])

        # Obsługa brakujących wartości w innych kolumnach
        data = data.dropna(subset=['Spam/Ham', 'Date'])  # Usunięcie wierszy bez tych kluczowych danych

        # Konwersja daty na format datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Obsługa błędów parsowania dat
        data = data.dropna(subset=['Date'])

        # Konwersja kolumny 'Spam/Ham' do formatu binarnego
        data['Spam'] = data['Spam/Ham'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

        # Usunięcie kolumny 'Spam/Ham' (opcjonalnie)
        data = data.drop(columns=['Spam/Ham'])

        print("Przetworzone dane:")
        print(data.head())
        print(".")
        print(".")
        print(".")
        print(data.tail())

        return data

    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None

if __name__ == "__main__":
    csv_path = "../dataset/emails.csv"  
    processed_data = load_and_clean_data(csv_path)
