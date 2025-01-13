"""
Metody klasyfikacji zaimplementowane jako algorytmy
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def decision_tree_classifier(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, **kwargs):
    """
    Klasyfikator Decision Tree z parametrami
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        max_depth: Maksymalna głębokość drzewa
        min_samples_split: Minimalna liczba próbek wymagana do podziału węzła
        **kwargs: Dodatkowe parametry
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu 
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    # 2. Trenowanie modelu
    clf.fit(X_train, y_train)

    # 3. Przewidywanie etykiet
    y_pred = clf.predict(X_test)

    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }

    return results

def naive_bayes_classifier(X_train, X_test, y_train, y_test, alpha=1.0, **kwargs):
    """
    Klasyfikator Naive Bayes z parametrami
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        alpha: Parametr wygładzania (Laplace/Lidstone)
        **kwargs: Dodatkowe parametry
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu 
    nb = MultinomialNB(alpha=alpha)

    # 2. Trenowanie modelu
    nb.fit(X_train, y_train)

    # 3. Przewidywanie etykiet
    y_pred = nb.predict(X_test)

    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }

    return results

def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5, **kwargs):
    """
    Klasyfikator K-Nearest Neighbors z parametrami
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        n_neighbors: Liczba najbliższych sąsiadów
        **kwargs: Dodatkowe parametry
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu 
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # 2. Trenowanie modelu
    knn.fit(X_train, y_train)
    
    # 3. Przewidywanie etykiet dla danych testowych
    y_pred = knn.predict(X_test)
    
    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }
    
    return results

def svm_classifier(X_train, X_test, y_train, y_test, kernel="linear", C=1.0, **kwargs):
    """
    Klasyfikator Support Vector Machines z parametrami
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        kernel: Funkcja jądra
        C: Parametr regularyzacji
        **kwargs: Dodatkowe parametry
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu
    svm = SVC(kernel=kernel, C=C, random_state=42)
    
    # 2. Trenowanie modelu
    svm.fit(X_train, y_train)
    
    # 3. Przewidywanie etykiet dla danych testowych
    y_pred = svm.predict(X_test)
    
    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }
    
    return results

def neural_network_classifier(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,), max_iter=300, **kwargs):
    """
    Klasyfikator Neural Network (Multi-Layer Perceptron) z parametrami

    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        hidden_layer_sizes: Rozmiary warstw ukrytych
        max_iter: Maksymalna liczba iteracji
        **kwargs: Dodatkowe parametry

    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # Inicjalizacja i trenowanie modelu z parametrami
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    # Przewidywanie etykiet
    y_pred = mlp.predict(X_test)

    # Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }

    return results
