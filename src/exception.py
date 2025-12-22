import sys
import logging

# Function to format the error message with details like file name, line number, and the actual error
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get traceback details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the name of the file where the error occurred
    line_number = exc_tb.tb_lineno  # Get the line number where the error occurred
    error_message = f"Error occurred in script [{file_name}] at line number [{line_number}] with error message [{str(error)}]"
    return error_message


# Custom exception class that inherits from the base Exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Call the base class constructor with the error message
        super().__init__(error_message)
        # Set the error message by calling the helper function
        self.error_message = error_message_detail(error_message, error_detail)

    # Override the __str__ method to return the formatted error message
    def __str__(self):
        return self.error_message


# # Main block to test the exception handling
# if __name__ == "__main__":
#     # Setting up logging configuration
#     logging.basicConfig(
#         filename="error_log.log",  # Log to file instead of console
#         format="[%(asctime)s] [Line: %(lineno)d] [%(levelname)s] - %(message)s",
#         level=logging.INFO,  # Set log level to INFO
#     )

#     try:
#         # This will raise a division by zero error
#         a = 1 / 0
#     except Exception as e:
#         # Log the error message before raising the custom exception
#         logging.error("Divide by Zero error occurred!")
#         # Raise the custom exception with the original error and traceback info
#         raise CustomException(e, sys)
