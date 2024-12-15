import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from classification.data_processing import load_data, preprocess_data, split_data
from classification.feature_extraction import extract_features
from classification.models import train_decision_tree, train_naive_bayes, train_knn, train_svm, train_neural_network

def main():
    # Wczytanie danych
    file_path = '../dataset/enron_spam_data.csv'  
    data = load_data(file_path)

    # Przetwarzanie danych
    preprocessed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)

    # Ekstrakcja cech
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Trening modeli
    models = {
        'Decision Tree': train_decision_tree(X_train_features, y_train),
        'Naive Bayes': train_naive_bayes(X_train_features, y_train),
        'K-Nearest Neighbors': train_knn(X_train_features, y_train),
        'Support Vector Machines': train_svm(X_train_features, y_train),
        'Neural Network': train_neural_network(X_train_features, y_train)
    }

    # Porównanie wyników
    for model_name, model in models.items():
        y_pred = model.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} - Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
