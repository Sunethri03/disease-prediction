# Disease Prediction App 🩺

A Machine Learning application that predicts potential diseases based on user-selected symptoms. Built with Python, this project provides both a desktop interface (Tkinter) and a web interface (Streamlit) to easily predict diseases using a Naive Bayes classifier.

## Features ✨

* **Symptom Selection**: Users can select up to 5 symptoms from a comprehensive list of 132 different symptoms.
* **Disease Prediction**: The model predicts over 41 different diseases, ranging from common allergies to more serious conditions like Malaria, Dengue, or Hypertension.
* **Two Interfaces**: Choose to run the application either as a Desktop App (`main.py`) or as a Modern Web App (`app.py`).

## Dataset 📊

The model is trained on a dataset (`Training.csv`) consisting of various symptoms mapped to specific diseases, and its accuracy is validated against `Testing.csv`.

## Prerequisites 🛠️

Ensure you have Python installed. You will need the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `streamlit`

You can install the dependencies by running:
```bash
pip install numpy pandas scikit-learn streamlit
```

## How to Run 🚀

### 1. Web Application (Streamlit)
To run the modern web interface:
```bash
streamlit run app.py
```
This will start a local server, and you can access the application in your browser at `http://localhost:8501`.

### 2. Desktop Application (Tkinter)
To run the standard desktop graphical interface:
```bash
python main.py
```
This will open a separate window where you can select your symptoms and predict the disease.

## How it Works 🧠

1. **Preprocessing**: The selected symptoms are converted into a binary array against the list of 132 known symptoms.
2. **Model**: A `MultinomialNB` (Multinomial Naive Bayes) algorithm is trained on `Training.csv`.
3. **Prediction**: The binary array of user symptoms is passed to the trained model to predict the most probable disease.

## Disclaimer ⚠️
This application is for educational purposes only and is not meant to replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
