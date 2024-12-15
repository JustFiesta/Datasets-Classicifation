from flask import Flask
from flaskr.routes import setup_routes

def create_app():
    """Funkcja tworząca i konfigurująca aplikację Flask"""
    app = Flask(__name__)

    # Dodanie tras
    setup_routes(app)

    return app

def main():
    """Uruchomienie serwera deweloperskiego"""
    app = create_app()

    # Tryb debugowania włączony tylko dla środowiska deweloperskiego
    app.run(
        host='0.0.0.0',  # Nasłuchiwanie na wszystkich interfejsach
        port=5000,        # Domyślny port
        debug=True        # Tryb debugowania
    )

if __name__ == '__main__':
    main()
