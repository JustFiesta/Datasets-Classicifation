import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

class EmailClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True),
            'Neural Network': MLPClassifier(max_iter=1000)
        }

    def preprocess_data(self, features_columns, target_column):
        """Przygotowanie danych do treningu"""
        self.X = self.data[features_columns]
        self.y = self.data[target_column]

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_models(self):
        """Trening wszystkich modeli"""
        results = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            results[name] = {
                'accuracy': model.score(self.X_test, self.y_test),
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }

            # Zapis modelu
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{name.replace(" ", "_")}_model.pkl')

        return results
