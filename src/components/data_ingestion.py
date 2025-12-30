# Importing necessary libraries for file operations, logging, data manipulation, and model components
import os
import sys
from src.exception import CustomException  # Importing custom exception class for error handling
from src.logger import logging  # Importing logging module for logging information
import pandas as pd  # Pandas for data manipulation and reading CSV files

# Importing methods for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing dataclass for configuration storage
from dataclasses import dataclass

# Importing custom components for data transformation and model training
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# A dataclass to store configuration for data ingestion
@dataclass
class DataIngestionConfig:
    # Paths for storing training, testing, and raw data
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# DataIngestion class for handling data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize the DataIngestionConfig object to store file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log that the data ingestion method has started
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the raw dataset from a CSV file using pandas
            df = pd.read_csv(r'C:\Users\Laksh\Videos\DS_AI\4-ML\4-ML\4-Machine Learning\24-End To End ML Project With Deployment\mlproject\notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for saving files exists, if not, create it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset as a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")

            # Split the dataset into training and test sets (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set and test set to respective files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths of the training and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If any exception occurs, raise a custom exception
            raise CustomException(e, sys)

# The main execution point of the script
if __name__ == "__main__":
    # Initialize the DataIngestion object and start the ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Initialize the DataTransformation object to preprocess the data
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Initialize the ModelTrainer object and start model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
