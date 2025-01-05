import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load("model.pkl")

# Example input data (replace with actual test data)
input_data = pd.DataFrame(
    {
        "latitude": [37.88],
        "housing_median_age": [41.0],
        "total_rooms": [880.0],
        "median_income": [8.3252],
        "ocean_proximity_INLAND": [0],  # 'No'
        "ocean_proximity_NEAR BAY": [1],  # 'No'
        "ocean_proximity_NEAR OCEAN": [0],  # 'No'
    }
)

# Scaler used during training (you need to re-fit it on training data, but we assume it's available here)
scaler = StandardScaler()

# Apply the same scaling as during training
# Assuming the same scaler was used in training and is now fitted
# Scale numerical features (this must match the features used in training)
num_features = [
    "latitude",
    "housing_median_age",
    "total_rooms",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]

# Fit the scaler on the training data or load it if saved separately
# Here we'll use a new scaler fit on the test data, as we don't have access to the original
input_data[num_features] = scaler.fit_transform(input_data[num_features])

# Make a prediction with the model
predicted_value = model.predict(input_data)

# Print the predicted value
print("Predicted Median House Value: $", predicted_value[0])

# Actual value (from your test data or known ground truth)
# Replace with actual value for comparison
actual_value = 452600  # Example actual value, replace with the true value

# Compare prediction with actual value
print(f"Actual Median House Value: ${actual_value}")
print(f"Prediction Error: ${abs(predicted_value[0] - actual_value)}")
