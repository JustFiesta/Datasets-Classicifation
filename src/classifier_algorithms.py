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

def neural_network_classifier(data, method="Neural Network"):
    """
    Klasyfikator Neural Network (Multi-Layer Perceptron) przystosowany do danych tekstowych.

    Args:
        data (DataFrame): Dane zawierające tekst oraz etykiety (kolumny 'Combined_Message' i 'Spam').
        method (str): Nazwa metody klasyfikacji (domyślnie "Neural Network").

    Returns:
        dict: Słownik z wynikami klasyfikacji.
    """
    
        # Przetworzenie danych
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['Combined_Message']).toarray()
    y = data['Spam']

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie wybranego modelu
    if method == "Neural Network":
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)

        # Przewidywanie etykiet
        y_pred = mlp.predict(X_test)

        # Obliczanie metryk
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=1),
            "recall": recall_score(y_test, y_pred, zero_division=1),
            "f1_score": f1_score(y_test, y_pred, zero_division=1),
        }

        return results
