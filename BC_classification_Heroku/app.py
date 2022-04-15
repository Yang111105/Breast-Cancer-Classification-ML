# Import libraries
import numpy as np
from flask import Flask, request, render_template
from pickle import load

# from tensorflow.keras.models import load_model ... use this only for a neural network,
# which will NOT be supported during boot camp, due to compatibility issues between
# tensorflow, Heroku, and Jupyter notebooks. 

# Initialize the flask App
app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Helps to avoid cache problems

# Load the model from its pickle file. (This pickle 
# file was originally saved by the code that trained 
# the model. See mlmodel.py)
# model = load_model('nn_model.h5') ... use this only for a neural network. See above. 
model = load(open('logistic_model.pkl', 'rb'))

# Load the scaler from its pickle file. This pickle
# file was originally saved by the code that trained 
# the model. See mlmodel.py for more. 
scaler = load(open('scaler_cs.pkl','rb'))

# Define the index route
@app.route('/')
def home():
    return render_template('index.html')

# Define a route that runs when the user clicks the Predict button in the web-app
@app.route('/predict', methods=['POST'])
def predict():
    
    # Create a list of the output labels.
    prediction_labels = ['Benign', 'Malignant']
    
    # Read the list of user-entered values from the website. Note that these
    # will be strings so we'll convert them to floats. 
    features = [float(x) for x in request.form.values()]

    # Put the list of floats into another list, to make scikit-learn happy. 
  
    final_features = [np.array(features)]
    print(f'Data from website: {final_features}')
     
    # Preprocess the input using the ORIGINAL scaler. 
    final_features_scaled = scaler.transform(final_features)

    # Use the scaled values to make the prediction. 
    prediction_encoded = model.predict(final_features_scaled) 
    print(f'prediction encoded = {prediction_encoded}')
   
    prediction = prediction_labels[prediction_encoded[0]]

    # Render a template that shows the result.
    prediction_text = f'Tumor Type:  {prediction}'
    # prediction_text = f'This is my prediction'
    # features = [10,20,30]
    return render_template('index.html', prediction_text=prediction_text, features=features)


# Allow the Flask app to launch from the command line
if __name__ == "__main__":
    app.run(debug=True)