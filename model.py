import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

data = pd.read_csv("./housing.csv")

# Number of missing values of features
print(data.isnull().sum())

# Basic summary statistics of features
print(data.describe())

# Fill missing values with the median
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

# One-hot encoding of categorical variable
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

print(data.columns)

# Select all of the numerical features for scaling
num_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Scales numerical features to a similar range
data[num_features] = StandardScaler().fit_transform(data[num_features])

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()


# Define features and target variable
# X = data.drop(["median_house_value"], axis=1)
X = data[
    [
        "latitude",
        "housing_median_age",
        "total_rooms",
        "median_income",
        "ocean_proximity_INLAND",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN"
    ]
]
y = data["median_house_value"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate the model
predictions = model.predict(X_test)
print(predictions[0])
mse = mean_squared_error(y_test, predictions)
rmse = sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Save the model as a .pkl file to the current directory
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

print("Exiting")
