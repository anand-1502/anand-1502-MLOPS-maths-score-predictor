import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

# Function to save an object to a file (using joblib for efficiency with large objects)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Use joblib for saving large objects (e.g., models)
        joblib.dump(obj, file_path)
        print(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(f"Error while saving object to {file_path}: {str(e)}", sys)

# Function to evaluate models using GridSearchCV and return performance scores
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Update model with best parameters from GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 score for training and test datasets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score of the model in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(f"Error during model evaluation: {str(e)}", sys)

# Function to load an object from a file (use joblib for consistency)
def load_object(file_path):
    try:
        # Use joblib to load objects (models, preprocessors)
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(f"Error while loading object from {file_path}: {str(e)}", sys)
