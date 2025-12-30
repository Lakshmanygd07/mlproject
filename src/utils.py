# Importing standard libraries for system and file handling
import os
import sys

# Importing external libraries for data processing and model evaluation
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation
import dill  # Dill for serialization (not used in this code)
import pickle  # Pickle for serializing Python objects to files
from sklearn.metrics import r2_score  # r2_score to evaluate model accuracy (R-squared)
from sklearn.model_selection import GridSearchCV  # GridSearchCV for hyperparameter tuning

# Importing a custom exception class for handling errors
from src.exception import CustomException

# Function to save Python objects to a file
def save_object(file_path, obj):
    try:
        # Getting the directory path of the given file path
        dir_path = os.path.dirname(file_path)

        # Creating the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Opening the file in write-binary mode and serializing the object to it
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # If an error occurs, raise a custom exception with details
        raise CustomException(e, sys)

# Function to evaluate multiple models on training and testing data
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Dictionary to store the evaluation results for each model
        report = {}

        # Loop through each model to evaluate performance
        for i in range(len(models)):
            model_name = list(models.keys())[i]  # Get the model name
            model = list(models.values())[i]  # Get the actual model

            # Get the hyperparameter grid for the model if available
            para = param.get(model_name, {})

            # If hyperparameters are provided, perform grid search for best parameters
            if para:
                gs = GridSearchCV(model, para, cv=3)  # 3-fold cross-validation
                gs.fit(X_train, y_train)  # Fit the grid search to the training data
                model.set_params(**gs.best_params_)  # Set the model's parameters to the best ones found

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predicting on the training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating the R-squared score for both training and testing sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Storing the test score in the report dictionary
            report[model_name] = test_model_score

        # Return the report with model names and their respective test scores
        return report

    except Exception as e:
        # If an error occurs during model evaluation, raise a custom exception
        raise CustomException(e, sys)

# Function to load a serialized object from a file
def load_object(file_path):
    try:
        # Open the file in read-binary mode and load the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # If an error occurs, raise a custom exception with details
        raise CustomException(e, sys)
