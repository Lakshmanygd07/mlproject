# Importing necessary modules
from setuptools import find_packages, setup  # 'find_packages' is used to find all packages in the project
from typing import List  # 'List' is used to type hint the expected return type of the function

# Defining a constant for the special entry '-e .' (editable installation)
HYPEN_E_DOT = '-e .'

# Function to read the 'requirements.txt' file and return a list of dependencies
def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements from the specified file path.
    It reads the file and processes it to remove unnecessary new lines and the special entry '-e .'.
    '''
    requirements = []  # Initialize an empty list to store requirements
    
    # Open the requirements.txt file for reading
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()  # Read all lines in the file into a list
        
        # Remove the newline characters from each line
        requirements = [req.replace("\n", "") for req in requirements]
        
        # If the special entry '-e .' is found, remove it from the list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements  # Return the cleaned list of requirements

# Setup function to configure the packaging for the project
setup(
    name='mlproject',  # The name of the project (package)
    version='0.0.1',  # The version of the project
    author='lakshman',  # Author's name
    author_email='lakshman.ygd@gmail.com',  # Author's email address
    packages=find_packages(),  # Automatically find all packages in the project directory
    install_requires=get_requirements('requirements.txt')  # Install dependencies from 'requirements.txt'
)
