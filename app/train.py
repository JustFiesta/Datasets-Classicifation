from app.classification.models import EmailClassifier

def main():
    # Przykładowe użycie - dostosuj do swoich danych
    classifier = EmailClassifier('data/emails.csv')

    # Przykładowe nazwy kolumn - dostosuj do swojego zbioru danych
    classifier.preprocess_data(
        features_columns=['feature1', 'feature2', 'feature3'],
        target_column='category'
    )

    results = classifier.train_models()

    # Wyświetlenie wyników treningu
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']}")
        print("Classification Report:\n", metrics['classification_report'])

if __name__ == '__main__':
    main()
