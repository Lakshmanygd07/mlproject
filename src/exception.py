import sys
from src.logger import logging  # Importing the logging module from the 'src' package

# Function to format the error message with details such as file name, line number, and the error message itself
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get the traceback details from the exception
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract the file name where the error occurred
    # Format the error message with file name, line number, and the error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


# Custom exception class that inherits from the base Exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call the parent Exception constructor with the error message
        # Set the error message by calling the error_message_detail function to format the message
        self.error_message = error_message_detail(error_message, error_detail)

    # Override the __str__ method to return the formatted error message when the exception is raised
    def __str__(self):
        return self.error_message


# Main block to check if the CustomException is working correctly
if __name__ == "__main__":

    try:
        # This will raise a ZeroDivisionError (attempting to divide by zero)
        a = 1 / 0
    except Exception as e:
        # Log the error message indicating a "Divide by Zero" error
        logging.info("Divide by Zero")  # Use info level for logging the occurrence
        # Raise the custom exception, passing the original error and traceback details
        raise CustomException(e, sys)
