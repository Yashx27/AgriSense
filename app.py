from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('/Users/yashkuchhal/My Files/IBM Internship 2/Crop Recommendation System/AgriSense/agrisense_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form on the web page
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Prepare the input array for prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict the crop
    prediction = model.predict(features)
    crop_recommendation = prediction[0]
    
    # Return the prediction result to the user
    return render_template('index.html', prediction_text=f'Recommended Crop: {crop_recommendation}')

if __name__ == "__main__":
    app.run(debug=True)