import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import numpy as np

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data
def preprocess_data(data):
    X = data['text']
    y = data['label']
    return X, y

# Function to augment data with patterns
def augment_data_with_patterns(X, y, patterns):
    for label, pattern_list in patterns.items():
        X += pattern_list
        y += [label] * len(pattern_list)
    return X, y

# Function to create and train the model
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svc', SVC(kernel='linear', probability=True))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Function to evaluate the model
def evaluate_model(pipeline, X_test, y_test, target_names):
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Function to save the model
def save_model(pipeline, file_path):
    joblib.dump(pipeline, file_path)

# Function to load the model
def load_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

def main():
    data_path = 'data/dataset.csv'
    model_path = 'models/model.pkl'
    pattern_file_path = 'patterns/patterns.json'

    # Load data
    data = load_data(data_path)
    X, y = preprocess_data(data)

    # Load patterns from JSON file
    with open(pattern_file_path) as f:
        patterns = json.load(f)

    # Augment data with patterns
    X, y = augment_data_with_patterns(X.tolist(), y.tolist(), patterns)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Load or train the model
    model = load_model(model_path)
    if model is None:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test, le.classes_)

    else:
        print("Model loaded from disk.")

    # Input loop
    while True:
        new_text = input("Enter the new input text (press 'q' to quit): ")
        if new_text.lower() == 'q':
            break

        predicted_label = model.predict([new_text])[0]
        predicted_label = le.inverse_transform([predicted_label])[0]
        print(f"Predicted label: {predicted_label}")

        # Ask for confirmation
        confirmation = input("Is this classification correct? (yes/no): ").lower()
        if confirmation == 'yes':
            # Add user's input to the training data and retrain the model
            user_label = input("Is it a hardware (HW) or software (SW) problem? ").strip().upper()
            if user_label in ['HW', 'SW']:
                # Check if the new label is unseen
                if user_label not in le.classes_:
                    # Add the new label to the classes and transform y again
                    le.classes_ = np.append(le.classes_, user_label)
                    y_encoded = le.transform(y)

                # Retrain the model with updated data
                model = train_model(X, y_encoded)
                save_model(model, model_path)

                print("Model retrained with user input.")
                print("Thank you for your feedback! The model has been updated.")

        elif confirmation == 'no':
            # Add user's input to the training data and retrain the model
            user_label = input("Is it a hardware (HW) or software (SW) problem? ").strip().upper()
            if user_label in ['HW', 'SW']:
                # Check if the new label is unseen
                if user_label not in le.classes_:
                    # Add the new label to the classes and transform y again
                    le.classes_ = np.append(le.classes_, user_label)
                    y_encoded = le.transform(y)

                # Retrain the model with updated data
                model = train_model(X, y_encoded)
                save_model(model, model_path)

                print("Model retrained with user input.")
                print("Thank you for your feedback! The model has been updated.")

        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


if __name__ == "__main__":
    main()
