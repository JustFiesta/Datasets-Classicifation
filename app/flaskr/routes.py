import os
import joblib
from flask import render_template, request, jsonify
from app.classification.models import EmailClassifier

def setup_routes(app):
    @app.route('/')
    def index():
        name = "Flask User"
        return render_template("index.html", name=name)
        # return "Hello world"

    @app.route('/classify', methods=['POST'])
    def classify_email():
        # Logika klasyfikacji konkretnego emaila
        email_text = request.form.get('email_text')

        # Tutaj dodasz logikę przygotowania tekstu do klasyfikacji
        # np. wektoryzacja, preprocessing

        results = {}
        model_files = os.listdir('models')

        for model_file in model_files:
            model = joblib.load(os.path.join('models', model_file))
            # Przewidywanie kategorii
            prediction = model.predict([email_text])
            results[model_file.replace('_model.pkl', '')] = prediction[0]

        return render_template('results.html', results=results)
