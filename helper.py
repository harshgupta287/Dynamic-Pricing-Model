import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(data):
    """Preprocess the dataset by handling missing values and encoding categorical variables."""
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

    categorical_features = data.select_dtypes(include=['object']).columns
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    # Convert categorical feature "Vehicle_Type" to one-hot encoded columns
    data = pd.get_dummies(data, columns=["Vehicle_Type"], drop_first=True)

    return data


def train_model(data):
    """Train a RandomForestRegressor model on the processed dataset."""
    model = RandomForestRegressor()
    
    x = data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type_Premium", "Expected_Ride_Duration"]]
    
    if "adjusted_ride_cost" in data.columns:
        y = data["adjusted_ride_cost"]
        model.fit(x, y)
        return model
    else:
        return None


def predict_price(model, number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration):
    """Predict ride cost based on user inputs."""
    vehicle_type_numeric = 1 if vehicle_type == "Premium" else 0
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
    predicted_price = model.predict(input_data)
    return predicted_price
