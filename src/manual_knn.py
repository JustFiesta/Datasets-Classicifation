from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

class ManualKNN:
    def __init__(self, k=3, weights='uniform', metric='euclidean', p=2):
        """
        Inicjalizuj klasyfikator KNN z parametrami
        
        Args:
            k (int): Liczba sąsiadów
            weights (str): Funkcja wag {'uniform', 'distance'}
            metric (str): Metryka odległości {'euclidean', 'manhattan', 'minkowski'}
            p (int): Parametr potęgi dla metryki Minkowskiego
        """
        self.k = k
        self.weights = weights
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def _calculate_weights(self, distances):
        """Oblicz wagi dla sąsiadów na podstawie ich odległości"""
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        elif self.weights == 'distance':
            # Dodaj małą stałą, aby uniknąć dzielenia przez zero
            return 1 / (distances + 1e-10)
        else:
            raise ValueError(f"Nieznana funkcja wag: {self.weights}")
    
    def _calculate_distances(self, X1, X2):
        """Oblicz odległości między punktami używając określonej metryki"""
        if self.metric == 'euclidean':
            # Odległość euklidesowa przy użyciu operacji macierzowych
            a2 = np.sum(X1**2, axis=1, keepdims=True)
            b2 = np.sum(X2**2, axis=1)
            ab = np.dot(X1, X2.T)
            distances = np.sqrt(np.maximum(a2 + b2 - 2*ab, 0))
        elif self.metric == 'manhattan':
            # Odległość Manhattan
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.sum(np.abs(X2 - X1[i]), axis=1)
        elif self.metric == 'minkowski':
            # Odległość Minkowskiego
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.power(np.sum(np.power(np.abs(X2 - X1[i]), self.p), axis=1), 1/self.p)
        else:
            raise ValueError(f"Nieznana metryka: {self.metric}")
        return distances
    
    def fit(self, X_train, y_train):
        """Przechowuj dane treningowe jako np.array"""
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
    
    def predict_batch(self, X_test, batch_size=100):
        """
        Przewiduj etykiety dla punktów testowych używając operacji wektorowych
        
        Args:
            X_test (array-like): Dane testowe
            batch_size (int): Rozmiar partii dla efektywności pamięciowej
        """
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch = X_test[i:end_idx]
            
            # Oblicz odległości używając określonej metryki
            distances = self._calculate_distances(batch, self.X_train)
            
            # Znajdź k najbliższych sąsiadów
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_distances = np.take_along_axis(distances, k_indices, axis=1)
            
            # Oblicz wagi
            weights = self._calculate_weights(k_distances)
            
            # Głosowanie ważone dla każdego punktu w partii
            for j, (indices, point_weights) in enumerate(zip(k_indices, weights)):
                labels = self.y_train[indices]
                if self.weights == 'uniform':
                    predictions[i + j] = Counter(labels).most_common(1)[0][0]
                else:
                    # Głosowanie ważone
                    weighted_votes = {}
                    for label, weight in zip(labels, point_weights):
                        weighted_votes[label] = weighted_votes.get(label, 0) + weight
                    predictions[i + j] = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        return predictions
    
    def predict(self, X):
        """Przewiduj etykiety dla wszystkich punktów testowych"""
        X = np.asarray(X)
        return self.predict_batch(X)

def manual_knn_classifier(X_train, X_test, y_train, y_test, **kwargs):
    """
    Funkcja opakowująca dla ręcznej implementacji KNN
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        **kwargs: Dodatkowe parametry dla klasyfikatora KNN
            - n_neighbors (int): Liczba sąsiadów
            - weights (str): Funkcja wag {'uniform', 'distance'}
            - metric (str): Metryka odległości {'euclidean', 'manhattan', 'minkowski'}
            - p (int): Parametr potęgi dla metryki Minkowskiego
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # Wyciągnij parametry z kwargs z domyślnymi wartościami
    n_neighbors = kwargs.get('n_neighbors', 5)
    weights = kwargs.get('weights', 'uniform')
    metric = kwargs.get('metric', 'euclidean')
    p = kwargs.get('p', 2)
    
    # Konwertuj dane na float32 dla lepszej wydajności
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Normalizuj dane dla lepszej stabilności numerycznej
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)
    
    # Inicjalizuj i trenuj model
    knn = ManualKNN(k=n_neighbors, weights=weights, metric=metric, p=p)
    knn.fit(X_train, y_train)
    
    # Przewiduj etykiety
    y_pred = knn.predict(X_test)
    
    # Oblicz metryki
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }
    
    return results