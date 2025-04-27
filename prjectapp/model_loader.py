# ===============================
# 1. Import all libraries
# ===============================
import gzip
import pickle
import xgboost as xgb
import numpy as np
import os

# accident_prediction\prjectapp\model\final_xgb_model_small.pkl.gz
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_xgb_model_small.pkl.gz")

# ===============================
# 2. Load the Saved Model
# ===============================
with gzip.open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ===============================
# 3. Define the Feature Names
# ===============================
feature_names = [
    "Year", "Start_Lat", "Start_Lng", "Distance(mi)", 
    "Street", "City", "County", "State", "Airport_Code", 
    "Temperature(F)", "Wind_Chill(F)", "Visibility(mi)", 
    "Wind_Direction", "Weather_Condition", "Traffic_Signal", 
    "Sunrise_Sunset", "TimeDiff"
]

# ===============================
# 4. Define the Prediction Function
# ===============================
def predict_severity(input_features):
    try:
        # Convert to numpy array
        input_features = np.array(input_features).reshape(1, -1)

        # Check if input feature count matches
        if input_features.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, but got {input_features.shape[1]}")
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(input_features, feature_names=feature_names, enable_categorical=True)

        # Predict
        probabilities = model.predict(dmatrix)[0]

        # Prepare readable output
        severity_levels = ["Minor (1)", "Moderate (2)", "Serious (3)", "Severe (4)"]
        probability_dict = {severity_levels[i]: f"{probabilities[i] * 100:.2f}%" for i in range(len(probabilities))}
        
        return probability_dict

    except Exception as e:
        return {"error": str(e)}

# # ===============================
# # 5. Example: Predict for New Input
# # ===============================

# # New sample input (17 features)
# new_input = [2, 40.023335, -76.432617, 0.0, 35681, 3658, 679, 36, 962, 46.0, 39.6, 8.0, 2, 40, 0, 0, 29.816667]


# # Call the prediction function
# result = predict_severity(new_input)

# print("\n🔮 Prediction Output:")
# print(result)

# ===============================
# 6. You can predict again for another new input similarly
# ===============================

another_input = [0, 33.615738, -83.920578, 0.0, 19643, 1514, 872, 9, 64, 73.0, 73.0, 7.0, 0, 8, 0, 1, 45.0]

result2 = predict_severity(another_input)

print("\n🔮 Another Prediction Output:")
print(result2)
