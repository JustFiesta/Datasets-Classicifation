# Ten prompt pomoże zacząć od miejsca w obecnym stanie implementacji logiki

Potrzebuje pomocy przy tworzeniu projektu:
"
Project: Email Classification - Algorithm Comparison
Technologies:
Python
Scikit-learn
Streamlit
Goal:
Compare the accuracy of 5 email classification algorithms:
Decision Tree
Naive Bayes
K-Nearest Neighbors
Support Vector Machines
Neural Networks
Data:
CSV columns:
Message ID
Subject
Message
Spam/Ham (labels: spam, offer, not spam)
Date
Current Progress:
Data Preprocessing:

Successfully loaded and cleaned the CSV dataset.
Combined 'Subject' and 'Message' into a single 'Text' column for classification.
Split data into training and testing sets (using 80% for training, 20% for testing).
Project Structure:

Main files:
data_processing.py: Handles loading, cleaning, and splitting the data.
feature_extraction.py: To implement text feature extraction (e.g., TF-IDF).
models.py: Placeholder for model implementations and evaluation.
main.py: Main script to run the classification experiment.

Implement feature extraction (e.g., using TfidfVectorizer).
Train the five algorithms (Decision Tree, Naive Bayes, etc.).
Compare model performance (accuracy, precision, recall).
Integrate the models with the Flask web app for visualizing results.
Challenges:
Deciding on the best preprocessing techniques (e.g., stemming/lemmatization).
Measuring and comparing the accuracy of models effectively.

UWAGI:
- Projekt ma charakter badawczy
- Celem jest porównanie skuteczności algorytmów
- Nie modyfikujemy oryginalnych danych
"

## Dodatek na koniec

Można i trzeba korzystać z dostępnych bibliotek w celu ułatwienia prac.
Zawsze możesz dopytywać się o niejasności.
Proponuje podejście iteracyjne - wolę zrozumiały kod, który można udoskonalić po kolejnych wiadomościach.

## Ten prompt pozwoli wygenerować coś podobnego jak to wyżej

napisz prompt sumujący obecny stan prac nad projektem tak by umożliwić wznowienie prac nawet w wypadku utraty twojej pamięci
