# Importing necessary libraries for Flask web application, data processing, and prediction
from flask import Flask, request, render_template  # Flask components for handling routes and rendering templates
import numpy as np  # For numerical operations (though not directly used here)
import pandas as pd  # For data manipulation (though not directly used here)

# Importing necessary components for prediction pipeline
from sklearn.preprocessing import StandardScaler  # For scaling numerical features (though not used directly here)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Custom data class and prediction pipeline

# Initialize the Flask application
application = Flask(__name__)  # Create a Flask application instance

# Alias for the application
app = application

## Route for the homepage
@app.route('/')
def index():
    # Render the home page (HTML file) when the user visits the root URL
    return render_template('index.html')

## Route for prediction page (handles GET and POST requests)
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If the request is GET, return the page for entering data
        return render_template('home.html')
    else:
        # If the request is POST (form submission), process the data and predict
        # Retrieve form data from the user input in the web page
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),  # Reading score comes from writing_score form input
            writing_score=float(request.form.get('reading_score'))  # Writing score comes from reading_score form input
        )

        # Convert the data into a DataFrame format for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # For debugging, print the data frame
        print("Before Prediction")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        # Make the prediction using the pipeline
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the results on the web page
        return render_template('home.html', results=results[0])  # Display the first result (assuming a single prediction)

# Run the Flask app on the default host and port
if __name__ == "__main__":
    app.run(host="0.0.0.0")
