"""
Application runner
"""
from src.processing import load_and_clean_data
from src.classifier_algorithms import decision_tree_classifier

# Decission Tree imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """
    Main function to run classification and training process, and pass it to UI
    """
    # Load data
    # Preprocess
    # Run Training
    # Run clasificators
    # Run UI with results csv

    csv_path = "dataset/enron_spam_data.csv"
    processed_data = load_and_clean_data(csv_path)

    # Decission Tree
    X = processed_data['Combined_Message']  # Cechy
    y = processed_data['Spam']              # Etykiety

    # Konwersja tekstu na reprezentację numeryczną
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    print("Running deccission tree classifier")
    decision_tree_results = decision_tree_classifier(X_train, X_test, y_train, y_test)
    print("Wyniki klasyfikacji Decision Tree:")
    print(decision_tree_results)


if __name__ == "__main__":
    main()
