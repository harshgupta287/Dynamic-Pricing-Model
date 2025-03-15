import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from helper import preprocess_data, train_model, predict_price

# Set page config
st.set_page_config(page_title="Dynamic Pricing Model", layout="wide")

# Title
st.title("Dynamic Pricing Model for Ride Fare Prediction")

# Load dataset
data = pd.read_csv("dynamic_pricing.csv")

# Calculate demand_multiplier based on percentile for high and low demand
high_demand_percentile = 75
low_demand_percentile = 25

data['demand_multiplier'] = np.where(
    data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
    data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
    data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

# Calculate supply_multiplier based on percentile for high and low supply
high_supply_percentile = 75
low_supply_percentile = 25

data['supply_multiplier'] = np.where(
    data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
    np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
    np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers'])

# Define price adjustment factors for high and low demand/supply
demand_threshold_low = 0.8
supply_threshold_high = 0.8

# Calculate adjusted_ride_cost for dynamic pricing
data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
    np.maximum(data['demand_multiplier'], demand_threshold_low) *
    np.maximum(data['supply_multiplier'], supply_threshold_high)
)

# Process the data
data = preprocess_data(data)

# Train model
model = train_model(data)

# Streamlit Sidebar UI
st.sidebar.markdown(
    "<h1 style='text-align: center; color: #00ADB5; font-size: 24px;'>User Input</h1>", 
    unsafe_allow_html=True
)


st.sidebar.markdown("#### Ride Demand & Supply")
user_number_of_riders = st.sidebar.slider("Number of Riders", 1, 100, 50)
user_number_of_drivers = st.sidebar.slider("Number of Drivers", 1, 100, 25)

st.sidebar.markdown("#### Ride Details")
user_vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Economy", "Premium"])
expected_ride_duration = st.sidebar.slider("Expected Ride Duration (minutes)", 5, 60, 30)



# Predict using user inputs
if model:
    predicted_price = predict_price(model, user_number_of_riders, user_number_of_drivers, user_vehicle_type, expected_ride_duration)
    
    # Styled box for predicted price
    st.markdown(
        f"""
        <div style="
            background-color: #e0e0e0; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center; 
            font-size: 26px; 
            font-weight: bold;
            color: #333333;
            border: 2px solid #b0b0b0;
            width: 50%;
            margin: auto;
        ">
            Predicted Price: ${predicted_price[0]:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Model training failed due to missing data.")


# Visualization
st.markdown("### Actual vs Predicted Values")

# Create scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["adjusted_ride_cost"], y=model.predict(data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type_Premium", "Expected_Ride_Duration"]]), mode='markers', name='Actual vs Predicted', marker=dict(color='blue', opacity=0.6)))
fig.add_trace(go.Scatter(x=[min(data["adjusted_ride_cost"]), max(data["adjusted_ride_cost"])], y=[min(data["adjusted_ride_cost"]), max(data["adjusted_ride_cost"])], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))

fig.update_layout(
    title='Actual vs Predicted Values', 
    xaxis_title='Actual Values', 
    yaxis_title='Predicted Values',
    template='plotly_dark'
)
st.plotly_chart(fig, use_container_width=True)

# Show complete dataset in Streamlit
st.markdown("### Dataset Overview")
st.dataframe(data, height=350)
