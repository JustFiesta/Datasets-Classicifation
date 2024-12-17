"""
Główny entry point projektu - uruchamia wszystkie algorytmy i zbiera dane do CSV
"""
from src.processing import load_and_clean_data
from src.classifier_algorithms import decision_tree_classifier, naive_bayes_classifier, knn_classifier, svm_classifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

def main():
    """
    Main - uruchamia wszystkie algorytmy i zbiera dane do CSV
    """
    # Load data
    # Preprocess
    # Split data to traits and labels
    # Run clasificators
    # Create csv
    # Run UI with results from csv

    csv_path = "dataset/enron_spam_data.csv"
    processed_data = load_and_clean_data(csv_path)

    X = processed_data['Combined_Message']  # Cechy
    y = processed_data['Spam']              # Etykiety

    # Konwersja tekstu na reprezentację numeryczną
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    # Decission Tree
    # print("Running deccission tree classifier")
    # results_dt = decision_tree_classifier(X_train, X_test, y_train, y_test)
    # print("Wyniki klasyfikacji Decision Tree:")
    # print(results_dt)

    # Native Bayess
    # results_nb = naive_bayes_classifier(X_train, X_test, y_train, y_test)
    # print("Wyniki klasyfikacji Naive Bayes:")
    # print(results_nb)

    # KNN
    # results_knn = knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5)
    # print("Wyniki klasyfikacji K-Nearest Neighbors:")
    # print(results_knn)

    # Test różnych wartości k - KNN
    # for k in range(1, 11):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    #     print(f"k={k}, Średnia dokładność: {scores.mean():.3f}")

    # SVM
    # results_svm = svm_classifier(X_train, X_test, y_train, y_test, kernel="rbf", C=1.0)
    # print("Wyniki klasyfikacji Support Vector Machines:")
    # print(results_svm)

    # Test różnych jąder - SVM TODO
    # for C in [0.1, 1, 10, 100]:
    #     results = svm_classifier(X_train, X_test, y_train, y_test, kernel="rbf", C=C)
    #     print(f"C={C}, Wyniki: {results}")

    # Sieć neuronowa
    # Skalowanie danych
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # Wywołanie klasyfikatora Neural Network - TODO nie działa w pełni
    # results_nn = neural_network_classifier(X_train_scaled, X_test_scaled, y_train, y_test)
    # print("Wyniki klasyfikacji Neural Network:")
    # print(results_nn)

    # TODO użycie threads do uruchamiania procesów

if __name__ == "__main__":
    main()
