import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
from functools import partial

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

from processing import load_and_clean_data
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
        return load_and_clean_data("dataset/enron_spam_data.csv")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Displaying data
def display_data(data):
    st.subheader("Data preview")
    st.write(data.head())

# Displaying classification results
def display_classification_metrics(y_true, y_pred, method_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    st.dataframe(metrics_df)

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

    This is a comparison of different classification methods.

    5 methods of classifying certain dataset (emails) are tested here with a simple monolithic Python + Streamlit app.

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
                
    Git-hub repository (https://github.com/JustFiesta/Datasets-Classicifation)            
    """)

def run_classifier(classifier_func, X_train, X_test, y_train, y_test, classifier_name, **kwargs):
    """
    Wykonuje klasyfikację i zwraca wyniki w ustandaryzowanym formacie
    """
    try:
        if classifier_name == "Neural Network":
            results = classifier_func(kwargs.get('data'))
        else:
            results = classifier_func(X_train, X_test, y_train, y_test, **kwargs)
        
        y_pred = results.get("y_pred")
        
        return {
            "name": classifier_name,
            "results": {
                "accuracy": results.get("accuracy"),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "y_pred": y_pred
            }
        }
    except Exception as e:
        st.error(f"Błąd podczas wykonywania {classifier_name}: {str(e)}")
        return None

# Benchmark
def benchmark(data):
    st.title("Benchmark and Comparison")

    # Przygotowanie danych
    with st.spinner('Preparing data...'):
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['Combined_Message']).toarray()
        y = data['Spam']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lista klasyfikatorów do uruchomienia
    classifiers = [
        {
            "func": decision_tree_classifier,
            "name": "Decision Tree",
            "kwargs": {}
        },
        {
            "func": naive_bayes_classifier,
            "name": "Naive Bayes",
            "kwargs": {}
        },
        {
            "func": knn_classifier,
            "name": "K-Nearest Neighbors",
            "kwargs": {"n_neighbors": 5}
        },
        {
            "func": svm_classifier,
            "name": "Support Vector Machines",
            "kwargs": {"kernel": "linear", "C": 1.0}
        },
        {
            "func": neural_network_classifier,
            "name": "Neural Network",
            "kwargs": {"data": data}
        }
    ]

    # Progress bar dla wszystkich operacji
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    benchmark_results = {}
    
    # Równoległe wykonywanie klasyfikatorów
    with st.spinner('Performing classification...'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Przygotowanie zadań
            future_to_classifier = {
                executor.submit(
                    run_classifier,
                    clf["func"],
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    clf["name"],
                    **clf["kwargs"]
                ): clf["name"] for clf in classifiers
            }

            # Zbieranie wyników w miarę ich ukończenia
            completed = 0
            for future in concurrent.futures.as_completed(future_to_classifier):
                completed += 1
                progress = completed / len(classifiers)
                progress_bar.progress(progress)
                
                classifier_name = future_to_classifier[future]
                status_text.text(f'Completed {classifier_name}...')
                
                try:
                    result = future.result()
                    if result:
                        benchmark_results[result["name"]] = result["results"]
                except Exception as e:
                    st.error(f"Error while executing {classifier_name}: {str(e)}")

    # Usuwanie komponentów postępu
    progress_bar.empty()
    status_text.empty()

    # Wyświetlanie wyników
    if benchmark_results:
        # Tworzenie DataFrame z wynikami
        results_df = pd.DataFrame.from_dict(
            {name: {k: v for k, v in data.items() if k != 'y_pred'} 
             for name, data in benchmark_results.items()},
            orient="index"
        )
        
        st.subheader("Benchmark Results")
        st.dataframe(results_df)

        # Wykresy porównawcze
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            st.subheader(f"{metric.capitalize()} Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=results_df.index, y=results_df[metric], palette="viridis", ax=ax)
            plt.title(f"{metric.capitalize()} of Different Classification Methods")
            plt.ylabel(metric.capitalize())
            plt.xlabel("Classifier")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

# Classify function
def classify(data, method):
    st.write(f"Performing classification for: {method}")

    with st.spinner('Preparing data...'):
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['Combined_Message']).toarray()
        y = data['Spam']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner('Performing classification...'):
        classifier_params = {
            "Decision Tree": (decision_tree_classifier, {}),
            "Naive Bayes": (naive_bayes_classifier, {}),
            "K-Nearest Neighbors": (knn_classifier, {"n_neighbors": 5}),
            "Support Vector Machines": (svm_classifier, {"kernel": "linear", "C": 1.0}),
            "Neural Network": (neural_network_classifier, {"data": data})
        }

        if method not in classifier_params:
            st.error(f"Method {method} is not implemented.")
            return

        classifier_func, kwargs = classifier_params[method]
        result = run_classifier(classifier_func, X_train, X_test, y_train, y_test, method, **kwargs)

        if result:
            st.subheader(f"Results for the method: {method}")
            display_classification_metrics(y_test, result["results"]["y_pred"], method)
# Main app
def main():
    st.set_page_config(
        layout="wide",
        page_title="Classification comparison",
    ) 
    set_button_style()

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
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
    with col7:
        if st.button("Benchmark"):
            selected_method = "Benchmark"

    data = load_data()

    if selected_method == "About":
        about_page()
    elif selected_method == "Benchmark" and data is not None:
        benchmark(data)
    elif selected_method and data is not None:
        st.title(f"Classification method: {selected_method}")
        classify(data, selected_method)
    elif data is not None:
        st.title("Comparison of email classification methods")
        display_data(data)

if __name__ == "__main__":
    main()
