import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# Load pre-trained model and scaler
model = joblib.load('trained_model.pkl')  # Replace with your actual model file
scaler = joblib.load('scaler.pkl')   # Replace with your scaler file

# Streamlit App
st.title("Network Intrusion Detection")

# User Input Form
with st.form("intrusion_form"):
    st.header("Enter Network Parameters")

    # Input fields
    duration = st.number_input("Duration (seconds)", min_value=0.0, step=0.1)
    protocol_type = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
    source_bytes = st.number_input("Source Bytes", min_value=0.0, step=0.1)
    destination_bytes = st.number_input("Destination Bytes", min_value=0.0, step=0.1)

    # Submit button
    submit = st.form_submit_button("Predict")

# Mapping protocol types to numeric values
protocol_mapping = {"TCP": 0, "UDP": 1, "ICMP": 2}

# Prediction logic
if submit:
    try:
        # Convert protocol type to numeric
        protocol_numeric = protocol_mapping[protocol_type]

        # Create feature array
        features = np.array([[duration, protocol_numeric, source_bytes, destination_bytes]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make predictions
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"Prediction: {'Intrusion' if prediction[0] == 1 else 'Normal'}")
        st.write(f"Confidence: {proba[1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
