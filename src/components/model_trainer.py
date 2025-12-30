# Importing necessary libraries for system, machine learning, and logging functionality
import os
import sys
from dataclasses import dataclass

# Importing machine learning models and evaluation metrics
from catboost import CatBoostRegressor  # CatBoost model for regression
from sklearn.ensemble import (
    AdaBoostRegressor,  # AdaBoost ensemble regressor
    GradientBoostingRegressor,  # Gradient Boosting regressor
    RandomForestRegressor,  # Random Forest regressor
)
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import r2_score  # Metric to evaluate model performance (R-squared)
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors regressor (not used in this code)
from sklearn.tree import DecisionTreeRegressor  # Decision Tree regressor
from xgboost import XGBRegressor  # XGBoost regressor

# Importing custom exceptions and logging modules from the project
from src.exception import CustomException
from src.logger import logging

# Importing utility functions from the project
from src.utils import save_object, evaluate_models

# A dataclass to store configuration for the model trainer, including the model file path
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Path to save the trained model

# ModelTrainer class to handle model training, evaluation, and saving
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initialize the model configuration

    # Method to initiate the model training process
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Splitting the input training and test data into features and target variables
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features of the training data
                train_array[:, -1],   # Target variable of the training data
                test_array[:, :-1],   # Features of the test data
                test_array[:, -1],    # Target variable of the test data
            )

            # Defining the models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids for each model for grid search optimization
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},  # No hyperparameters for Linear Regression
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                }
            }

            # Calling the utility function to evaluate models and return their performance scores
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
            )

            # Getting the best model score from the model report
            best_model_score = max(sorted(model_report.values()))

            # Retrieving the best model based on its score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # If the best model score is below 0.6, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Saving the best model to the specified file path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Making predictions with the best model on the test data
            predicted = best_model.predict(X_test)

            # Calculating R-squared score for model evaluation on test data
            r2_square = r2_score(y_test, predicted)

            # Returning the R-squared score
            return r2_square

        # Handling any exceptions that may occur during the model training process
        except Exception as e:
            raise CustomException(e, sys)
