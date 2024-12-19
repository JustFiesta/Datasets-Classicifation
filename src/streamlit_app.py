import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from processing import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from classifier_algorithms import (
    decision_tree_classifier,
    naive_bayes_classifier,
    knn_classifier,
    svm_classifier,
    neural_network_classifier
)

def set_button_style():
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():

    try:
        return load_and_clean_data("../dataset/enron_spam_data.csv")
        # return pd.read_csv("../dataset/enron_spam_data.csv")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Displaying data
def display_data(data):
    st.subheader("Data preview")
    st.write(data.head())

# Displaying classification results
def display_classification_metrics(y_true, y_pred, method_name):
    st.subheader(f"Results for the method: {method_name}")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion matrix - {method_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(plt)

# About
def about_page():
    st.title("About project")
    st.markdown("""
    ## Compare classification methods

    This is a comparasion of different classification methods.

    5 methods of classifying certain dataset (emails) is tested here with simple monolithic Python + Streamlit app.

    Classification bases on 2 classes:
    - Spam
    - Not spam

    It does not change the data, just perform analysis on given and preprocessed dataset.

    Based on [Enron Spam Dataset](https://github.com/MWiechmann/enron_spam_data).

    ## Classifying methods
    - Decision Tree
    - Naive Bayes
    - K-Nearest Neighbors
    - Support Vector Machines
    - Neural Network

    ## Used Technologies
    - Python
    - Streamlit
    - Scikit-learn
    """)

def classify(data, method):
    st.write(f"Starting the classification for: {method}")

    # Przetworzenie danych wejściowych
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['Combined_Message']).toarray()
    y = data['Spam']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wywołanie odpowiedniego klasyfikatora
    results = None
    y_pred = None
    if method == "Decision Tree":
        results = decision_tree_classifier(X_train, X_test, y_train, y_test)
        y_pred = results.get("y_pred")
    elif method == "Naive Bayes":
        results = naive_bayes_classifier(X_train, X_test, y_train, y_test)
        y_pred = results.get("y_pred")
    elif method == "K-Nearest Neighbors":
        results = knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5)
        y_pred = results.get("y_pred")
    elif method == "Support Vector Machines":
        results = svm_classifier(X_train, X_test, y_train, y_test, kernel="linear", C=1.0)
        y_pred = results.get("y_pred")
    elif method == "Neural Network":
        results = neural_network_classifier(data)
        y_pred = results.get("y_pred")
    else:
        st.error(f"Method {method} is not implemented.")
        return

    # Wyświetlenie wyników klasyfikacji
    if results:
        st.subheader(f"Results for the method: {method}")
        for metric, value in results.items():
            st.write(f"{metric.capitalize()}: {value:.4f}")

        # Wyświetlenie macierzy konfuzji
        if y_pred is not None:
            display_classification_metrics(y_test, y_pred, method)
def main():
    st.set_page_config(layout="wide") 
    set_button_style()

    # top buttons
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    selected_method = None
    with col1:
        if st.button("Decision Tree"):
            selected_method = "Decision Tree"
    with col2:
        if st.button("Naive Bayes"):
            selected_method = "Naive Bayes"
    with col3:
        if st.button("KNN"):
            selected_method = "K-Nearest Neighbors"
    with col4:
        if st.button("SVM"):
            selected_method = "Support Vector Machines"
    with col5:
        if st.button("Neural Network"):
            selected_method = "Neural Network"
    with col6:
        if st.button("About"):
            selected_method = "About"

    # load data
    data = load_data()

    if selected_method == "About":
        about_page()
    elif selected_method and data is not None:
        st.title(f"Classification method: {selected_method}")
        classify(data, selected_method)
    elif data is not None:
        st.title("Comparison of email classification methods")
        display_data(data)

if __name__ == "__main__":
    main()
