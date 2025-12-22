import logging
import os
from datetime import datetime

# Create a dynamic log file name using the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path to the logs folder
logs_path = os.path.join(os.getcwd(), "logs")

# Create the "logs" folder if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Define the full path to the log file, including the file name
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# # Check if logging is working by writing a test log entry
# if __name__ == "__main__":
#     logging.info("Logging has started")
