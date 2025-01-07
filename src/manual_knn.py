from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

class ManualKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """Zapamiętanie danych treningowych jako np.array"""
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
    
    def predict_batch(self, X_test, batch_size=100):
        """
        Przewidywanie etykiet dla grupy punktów testowych
        używając zoptymalizowanych operacji wektorowych
        """
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        
        # Przetwarzanie grupami dla oszczędności pamięci
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch = X_test[i:end_idx]
            
            # Obliczanie odległości wektorowo z zabezpieczeniem przed ujemnymi wartościami
            a2 = np.sum(batch**2, axis=1, keepdims=True)
            b2 = np.sum(self.X_train**2, axis=1)
            ab = np.dot(batch, self.X_train.T)
            
            # Zabezpieczenie przed ujemnymi wartościami pod pierwiastkiem
            distances_squared = np.maximum(a2 + b2 - 2*ab, 0)
            distances = np.sqrt(distances_squared)
            
            # Znajdowanie k najbliższych sąsiadów
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            
            # Głosowanie większościowe dla każdego punktu w grupie
            for j, indices in enumerate(k_indices):
                labels = self.y_train[indices]
                predictions[i + j] = Counter(labels).most_common(1)[0][0]
        
        return predictions
    
    def predict(self, X):
        """Przewidywanie etykiet dla wszystkich punktów testowych"""
        X = np.asarray(X)
        return self.predict_batch(X)
    
def manual_knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Funkcja wrapper dla manualnej implementacji KNN, dopasowana do interfejsu pozostałych klasyfikatorów
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        n_neighbors: Liczba sąsiadów (domyślnie 5)
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
# Konwersja danych do float32 dla lepszej wydajności
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Normalizacja danych dla lepszej stabilności numerycznej
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)
    
    # Inicjalizacja i trenowanie modelu
    knn = ManualKNN(k=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Przewidywanie etykiet
    y_pred = knn.predict(X_test)
    
    # Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "y_pred": y_pred,
    }
    
    return results