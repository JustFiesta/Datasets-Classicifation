from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(X_train, X_test):
    """Konwertuje tekst na cechy numeryczne."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec
