"""
Entry point do aplikacji - instaluje zależności i uruchamia interfejs Streamlit
"""
import subprocess
import sys
import os

def install_requirements():
    """
    Sprawdza i instaluje wymagane zależności z pliku requirements.txt
    """
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("Nie znaleziono pliku requirements.txt")
        return False
    
    print("Sprawdzanie i instalowanie zależności...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Wszystkie zależności zostały zainstalowane pomyślnie.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas instalacji zależności: {e}")
        return False

def main():
    """
    Główna funkcja instalująca zależności i uruchamiająca aplikację Streamlit
    """
    # Instalacja zależności
    if not install_requirements():
        print("Nie można kontynuować bez wymaganych zależności.")
        return

    # Ścieżka do pliku UI
    ui_path = os.path.join(os.path.dirname(__file__), "src", "streamlit_app.py")
    
    if not os.path.exists(ui_path):
        print(f"Błąd: Nie znaleziono pliku UI pod ścieżką: {ui_path}")
        return

    print("Uruchamianie aplikacji Streamlit...")
    
    # Uruchomienie aplikacji Streamlit
    try:
        subprocess.run(["streamlit", "run", ui_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas uruchamiania aplikacji Streamlit: {e}")
    except FileNotFoundError:
        print("Błąd: Nie można uruchomić Streamlit. Upewnij się, że jest prawidłowo zainstalowany.")

if __name__ == "__main__":
    main()