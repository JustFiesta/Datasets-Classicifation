"""
Metody klasyfikacji zaimplementowane jako algorytmy
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    """
    Klasyfikator Decision Tree
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu
    clf = DecisionTreeClassifier(random_state=42)

    # 2. Trenowanie modelu
    clf.fit(X_train, y_train)

    # 3. Przewidywanie etykiet
    y_pred = clf.predict(X_test)

    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return results

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    """
    Klasyfikator Naive Bayes
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu
    nb = MultinomialNB()

    # 2. Trenowanie modelu
    nb.fit(X_train, y_train)

    # 3. Przewidywanie etykiet
    y_pred = nb.predict(X_test)

    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return results

def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Klasyfikator K-Nearest Neighbors
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        n_neighbors: Liczba najbliższych sąsiadów (domyślnie 5)
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu z określoną liczbą sąsiadów
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # 2. Trenowanie modelu
    knn.fit(X_train, y_train)
    
    # 3. Przewidywanie etykiet dla danych testowych
    y_pred = knn.predict(X_test)
    
    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    
    return results

def svm_classifier(X_train, X_test, y_train, y_test, kernel="linear", C=1.0):
    """
    Klasyfikator Support Vector Machines
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        kernel: Funkcja jądra (domyślnie 'linear')
        C: Parametr regularyzacji (domyślnie 1.0)
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu SVM z wybraną funkcją jądra i parametrem C
    svm = SVC(kernel=kernel, C=C, random_state=42)
    
    # 2. Trenowanie modelu
    svm.fit(X_train, y_train)
    
    # 3. Przewidywanie etykiet dla danych testowych
    y_pred = svm.predict(X_test)
    
    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    
    return results

def neural_network_classifier(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,), max_iter=300):
    """
    Klasyfikator Neural Network (Multi-Layer Perceptron)
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
        hidden_layer_sizes: Rozmiar warstw ukrytych (domyślnie jedna warstwa z 100 neuronami)
        max_iter: Maksymalna liczba iteracji trenowania (domyślnie 300)
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    # 1. Inicjalizacja modelu MLP
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        activation='relu',
                        solver='adam',
                        random_state=42)

    # 2. Trenowanie modelu
    mlp.fit(X_train, y_train)

    # 3. Przewidywanie etykiet dla danych testowych
    y_pred = mlp.predict(X_test)

    # 4. Obliczanie metryk
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return results
