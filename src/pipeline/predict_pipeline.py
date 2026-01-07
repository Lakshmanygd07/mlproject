# Importing necessary libraries for handling exceptions, loading models, and data processing
import sys
import pandas as pd  # For creating and manipulating DataFrames
from src.exception import CustomException  # Custom exception class for error handling
from src.utils import load_object  # Utility function to load objects from files

# The PredictPipeline class is responsible for loading a pre-trained model and preprocessor,
# transforming input data, and making predictions.
class PredictPipeline:
    def __init__(self):
        # Constructor to initialize the class, no specific initialization required here.
        pass

    def predict(self, features):
        try:
            # Define the file paths for the model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # Log the message before loading the objects
            print("Before Loading")

            # Load the model and preprocessor using the custom `load_object` function
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log the message after loading the objects
            print("After Loading")

            # Apply the preprocessor (scaler, encoder, etc.) to the features
            data_scaled = preprocessor.transform(features)

            # Use the model to make predictions on the scaled features
            preds = model.predict(data_scaled)

            # Return the predictions
            return preds
        
        except Exception as e:
            # If any exception occurs during prediction, raise a custom exception
            raise CustomException(e, sys)

# The CustomData class is designed to collect user inputs, convert them to a DataFrame,
# and prepare them for prediction.
class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        # Initialize the user input data as instance variables
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # This method converts the instance variables into a pandas DataFrame for prediction
    def get_data_as_data_frame(self):
        try:
            # Create a dictionary from the instance variables
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Return the dictionary as a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # If any exception occurs during DataFrame creation, raise a custom exception
            raise CustomException(e, sys)
