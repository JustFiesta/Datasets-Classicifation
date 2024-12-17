"""
All classification methods implemented as functions
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    pass

def knn_classifier(X_train, X_test, y_train, y_test):
    """
    Klasyfikator K-Nearest Neighbors
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    pass

def svm_classifier(X_train, X_test, y_train, y_test):
    """
    Klasyfikator Support Vector Machines
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    pass

def neural_network_classifier(X_train, X_test, y_train, y_test):
    """
    Klasyfikator Neural Network
    
    Args:
        X_train, X_test: Dane treningowe i testowe
        y_train, y_test: Etykiety treningowe i testowe
    
    Returns:
        dict: Słownik z wynikami klasyfikacji
    """
    pass