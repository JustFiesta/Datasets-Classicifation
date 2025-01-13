import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import numpy as np
from functools import partial

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from processing_spam import load_and_clean_spam_data
from processing_mushrooms import load_and_clean_mushrooms_data
from classifier_algorithms import (
    decision_tree_classifier,
    naive_bayes_classifier,
    knn_classifier,
    svm_classifier,
    neural_network_classifier
)

from manual_knn import manual_knn_classifier

SPAM_DATA_PATH = "dataset/enron_spam_data.csv"

def set_button_style():
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(dataset_option):
    try:
        if dataset_option == 'Spam':
            return load_and_clean_spam_data(SPAM_DATA_PATH)
        else:
            return load_and_clean_mushrooms_data()
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
    
    plt.figure(figsize=(5, 4))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.title(f"Confusion matrix - {method_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    st.pyplot(plt, use_container_width=False)  

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

def get_algorithm_parameters(method):
    """Get user-defined parameters for each algorithm"""
    params = {}
    
    st.sidebar.subheader("Algorithm Parameters")
    
    if method == "Decision Tree":
        params["max_depth"] = st.sidebar.slider(
            "Maximum Depth",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum depth of the decision tree"
        )
        params["min_samples_split"] = st.sidebar.slider(
            "Minimum Samples Split",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum samples required to split an internal node"
        )
        
    elif method == "Naive Bayes":
        params["alpha"] = st.sidebar.slider(
            "Smoothing Parameter (Alpha)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Additive (Laplace/Lidstone) smoothing parameter"
        )
        
    elif method == "K-Nearest Neighbors":
        params["n_neighbors"] = st.sidebar.slider(
            "Number of Neighbors",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of neighbors to use for classification"
        )
        
    elif method == "Support Vector Machines":
        params["C"] = st.sidebar.slider(
            "Regularization Parameter (C)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Regularization parameter. Smaller values mean stronger regularization"
        )
        params["kernel"] = st.sidebar.selectbox(
            "Kernel Type",
            options=["linear", "rbf", "poly"],
            index=0,
            help="Type of kernel to use for the SVM"
        )
        
    elif method == "Neural Network":
        hidden_size = st.sidebar.slider(
            "Hidden Layer Size",
            min_value=10,
            max_value=200,
            value=100,
            help="Number of neurons in the hidden layer"
        )
        params["hidden_layer_sizes"] = (hidden_size,)
        params["max_iter"] = st.sidebar.slider(
            "Maximum Iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum number of iterations for training"
        )
    
    return params

def run_classifier(classifier_func, X_train, X_test, y_train, y_test, classifier_name, **kwargs):
    """
    Wykonuje klasyfikację i zwraca wyniki w ustandaryzowanym formacie
    """
    try:
        if classifier_name == "Neural Network":
            if isinstance(X_train, pd.DataFrame):  # Dla datasetu mushrooms
                # Łączymy dane treningowe i testowe z zresetowaniem indeksów
                X_train_reset = X_train.reset_index(drop=True)
                X_test_reset = X_test.reset_index(drop=True)
                X_combined = pd.concat([X_train_reset, X_test_reset], ignore_index=True)

                y_train_series = pd.Series(y_train).reset_index(drop=True)
                y_test_series = pd.Series(y_test).reset_index(drop=True)
                y_combined = pd.concat([y_train_series, y_test_series], ignore_index=True)

                # Przygotowanie pełnego datasetu z kolumną 'poisonous'
                full_data = X_combined.copy()
                full_data['poisonous'] = y_combined

                results = classifier_func(full_data)
            else:  # Dla datasetu spam
                # Łączymy dane treningowe i testowe
                X_combined = np.vstack((X_train, X_test))
                y_combined = np.concatenate((y_train, y_test))
  
                data = pd.DataFrame({
                    'Combined_Message': [' '.join(map(str, row)) for row in X_combined],
                    'Spam': y_combined
                })
                results = classifier_func(data)
        else:
            results = classifier_func(X_train, X_test, y_train, y_test, **kwargs)

        y_pred = results.get("y_pred")

        return {
            "name": classifier_name,
            "results": {
                "accuracy": results.get("accuracy"),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted'),
                "y_pred": y_pred
            }
        }
    except Exception as e:
        st.error(f"Błąd podczas wykonywania {classifier_name}: {str(e)}")
        return None

# Benchmark
def benchmark(data, dataset_option):
    st.title("Benchmark and Comparison")

    # Przygotowanie danych
    with st.spinner('Preparing data...'):
        if dataset_option == 'Spam':
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(data['Combined_Message']).toarray()
            y = data['Spam']
        else:
            X = data.drop(columns=['poisonous'])
            y = data['poisonous']
            le = LabelEncoder()
            X = X.apply(le.fit_transform)
            y = le.fit_transform(y)
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
            "func": manual_knn_classifier,
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
            "kwargs": {}
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
            
            fig, ax = plt.subplots(figsize=(5, 4))  
            
            sns.barplot(x=results_df.index, y=results_df[metric], palette="viridis", ax=ax)
            plt.title(f"{metric.capitalize()} of Different Classification Methods", fontsize=8)  
            plt.ylabel(metric.capitalize(), fontsize=8)  
            plt.xlabel("Classifier", fontsize=8)
            plt.xticks(rotation=45, fontsize=6)  
            plt.yticks(fontsize=6) 
            
            st.pyplot(fig, use_container_width=False) 
            plt.close(fig)


# Classify function
def classify(data, method, dataset_option):
    st.write(f"Performing classification for: {method}")
    
    if 'classification_started' not in st.session_state:
        st.session_state.classification_started = False
    
    params = get_algorithm_parameters(method)
    
    if st.button("Start Classification") or st.session_state.classification_started:
        st.session_state.classification_started = True
        
        with st.spinner('Preparing data...'):
            if dataset_option == 'Spam':
                vectorizer = TfidfVectorizer(max_features=1000)
                X = vectorizer.fit_transform(data['Combined_Message']).toarray()
                y = data['Spam']
            else:
                X = data.drop(columns=['poisonous'])
                y = data['poisonous']
                le = LabelEncoder()
                X = X.apply(le.fit_transform)
                y = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner('Performing classification...'):
            classifier_params = {
                "Decision Tree": (decision_tree_classifier, params),
                "Naive Bayes": (naive_bayes_classifier, params),
                "K-Nearest Neighbors": (manual_knn_classifier, params),
                "Support Vector Machines": (svm_classifier, params),
                "Neural Network": (neural_network_classifier, params)
            }

            if method not in classifier_params:
                st.error(f"Method {method} is not implemented.")
                return

            classifier_func, func_params = classifier_params[method]
            result = run_classifier(classifier_func, X_train, X_test, y_train, y_test, method, **func_params)

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

    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = None
    
    if 'classification_started' not in st.session_state:
        st.session_state.classification_started = False

    st.title("Dataset Classification App")

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

    def reset_classification():
        st.session_state.classification_started = False

    selected_method = None
    with col1:
        if st.button("Decision Tree"):
            st.session_state.selected_method = "Decision Tree"
            reset_classification()
    with col2:
        if st.button("Naive Bayes"):
            st.session_state.selected_method = "Naive Bayes"
            reset_classification()
    with col3:
        if st.button("KNN"):
            st.session_state.selected_method = "K-Nearest Neighbors"
            reset_classification()
    with col4:
        if st.button("SVM"):
            st.session_state.selected_method = "Support Vector Machines"
            reset_classification()
    with col5:
        if st.button("Neural Network"):
            st.session_state.selected_method = "Neural Network"
            reset_classification()
    with col6:
        if st.button("About"):
            st.session_state.selected_method = "About"
            reset_classification()
    with col7:
        if st.button("Benchmark"):
            st.session_state.selected_method = "Benchmark"
            reset_classification()

    # Create a selectbox for dataset selection
    dataset_option = st.selectbox(
        'Select the dataset to process',
        ('Spam', 'Mushrooms')
    )

    # Load and preprocess the selected dataset
    data = load_data(dataset_option)

    if st.session_state.selected_method == "About":
        about_page()
    elif st.session_state.selected_method == "Benchmark" and data is not None:
        benchmark(data, dataset_option)
    elif st.session_state.selected_method and data is not None:
        st.title(f"Classification method: {st.session_state.selected_method}")
        classify(data, st.session_state.selected_method, dataset_option)
    elif data is not None:
        st.write(f"Processed dataset {dataset_option}:")
        st.dataframe(data)

if __name__ == "__main__":
    main()