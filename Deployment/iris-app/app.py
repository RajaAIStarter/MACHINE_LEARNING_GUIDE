from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('iris_model.pkl')

# Create a Flask app
app = Flask(__name__)

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data as JSON
    data = request.json
    # Make predictions
    predictions = model.predict([data])
    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")