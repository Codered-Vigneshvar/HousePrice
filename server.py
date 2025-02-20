import pickle
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows requests from any origin

# Load the linear regression model
linear_model_path = "/Users/vigneshvars/Documents/MLLearn/linear/banglore_home_prices_model.pickle"
with open(linear_model_path, 'rb') as f:
    linear_model = pickle.load(f)

# Load the XGBoost model
xgb_model_path = "/Users/vigneshvars/Documents/MLLearn/linear/xgb_tuned_model.pickle"
with open(xgb_model_path, 'rb') as f:
    xgb_model = pickle.load(f)

# Load column names from columns.json
columns_path = "/Users/vigneshvars/Documents/MLLearn/linear/columns.json"
with open(columns_path, 'r') as f:
    data_columns_dict = json.load(f)
if isinstance(data_columns_dict, dict):
    data_columns = data_columns_dict.get("data_columns", [])
else:
    data_columns = data_columns_dict

# Endpoint to return location names (exclude the first three non-location columns)
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    locations = data_columns[3:]
    return jsonify({'locations': locations})

# Endpoint to predict price per sqft and total cost
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received payload:", data)
        total_sqft = float(data['total_sqft'])
        location = data['location']
        bath = int(data['bath'])
        bhk = int(data['bhk'])
        advanced = data.get('advanced', False)  # If true, use XGBoost; else, use linear regression

        # Prepare the feature vector
        x = np.zeros(len(data_columns))
        if "total_sqft" in data_columns:
            x[data_columns.index("total_sqft")] = total_sqft
        if "bath" in data_columns:
            x[data_columns.index("bath")] = bath
        if "bhk" in data_columns:
            x[data_columns.index("bhk")] = bhk

        # Set one-hot encoding for location
        if location in data_columns:
            x[data_columns.index(location)] = 1
        else:
            return jsonify({'error': f"Location '{location}' not found in model columns"}), 400

        # Use appropriate model based on the advanced flag
        if advanced:
            price_per_sqft = xgb_model.predict([x])[0]
        else:
            price_per_sqft = linear_model.predict([x])[0]

        total_cost = price_per_sqft * total_sqft

        result = {
            'price_per_sqft': float(price_per_sqft),
            'total_cost': float(total_cost)
        }
        print("Returning:", result)
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)



