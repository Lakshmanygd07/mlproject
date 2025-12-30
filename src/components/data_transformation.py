# Importing necessary libraries for data processing, transformation, and error handling
import sys
from dataclasses import dataclass

# Importing libraries for data manipulation and processing
import numpy as np  # For numerical operations
import pandas as pd  # For handling data in DataFrame format

# Importing scikit-learn modules for data preprocessing and building pipelines
from sklearn.compose import ColumnTransformer  # To apply different transformations to different columns
from sklearn.impute import SimpleImputer  # For filling missing values
from sklearn.pipeline import Pipeline  # To chain multiple steps into a pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding categorical data and scaling numerical data

# Importing custom exception and logging modules for error handling and logging
from src.exception import CustomException
from src.logger import logging

# Importing utility functions from the project
import os
from src.utils import save_object

# A dataclass to store the path for saving the preprocessing object
@dataclass
class DataTransformationConfig:
    # File path to save the preprocessing object
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# DataTransformation class to handle the data preprocessing and transformation
class DataTransformation:
    def __init__(self):
        # Initialize the configuration object for data transformation
        self.data_transformation_config = DataTransformationConfig()

    # This method builds and returns the data transformer object (pipeline of preprocessing steps)
    def get_data_transformer_object(self):
        '''
        This function is responsible for creating and returning the preprocessing pipeline
        that includes transformations for both numerical and categorical data.
        '''
        try:
            # List of numerical and categorical columns in the dataset
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Creating a pipeline for numerical data preprocessing
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with the median of the column
                    ("scaler", StandardScaler())  # Scale the numerical data to have mean=0 and std=1
                ]
            )

            # Creating a pipeline for categorical data preprocessing
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with the most frequent category
                    ("one_hot_encoder", OneHotEncoder()),  # One-hot encode the categorical columns
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical data, but do not center it (mean=False)
                ]
            )

            # Log the columns to be transformed
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline to numerical columns
                    ("cat_pipelines", cat_pipeline, categorical_columns)  # Apply categorical pipeline to categorical columns
                ]
            )

            # Return the preprocessor object (the combined pipeline)
            return preprocessor
        
        except Exception as e:
            # If an error occurs during preprocessing, raise a custom exception
            raise CustomException(e, sys)

    # This method initiates the data transformation process by reading data and applying transformations
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing data from CSV files into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log the completion of reading the datasets
            logging.info("Read train and test data completed")

            # Log the step of obtaining the preprocessing object
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing pipeline object
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name (the column to predict)
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split the DataFrame into features (input) and target (output)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Log that preprocessing will be applied to both the training and test data
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the preprocessing pipeline to the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the processed input features and target values into final transformed arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Log that the preprocessing object has been saved
            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing pipeline object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed training and test arrays and the file path of the saved preprocessing object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # If an error occurs during the data transformation process, raise a custom exception
            raise CustomException(e, sys)
