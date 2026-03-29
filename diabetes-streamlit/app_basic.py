import streamlit as st
import numpy as np
import dill as pickle

# Load the pre-trained model
with open('best_diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Assuming you have an image at 'path/to/logo.png'
logo_path = './logo_afi.png'

# Create columns for the logo and the title
col1, col2 = st.columns([1, 8])  # Adjust the ratio based on your layout preference

with col1:
    st.image(logo_path, width=500)  # Adjust the width as needed

# Streamlit webpage configuration
st.title('Diabetes Prediction Model')
st.write('Please enter values for each feature to get a diabetes progression prediction.')

# Define input fields for each feature of the diabetes dataset
feature_names = ['Age', 'Sex', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
user_input = []
default_values = [50.0, 0.0, 25.0, 75.0, 200.0, 150.0, 45.0, 5.0, 4.5, 85.0]  # Example default values

for feature, default in zip(feature_names, default_values):
    col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed
    with col1:
        st.write(f"{feature}:")
    with col2:
        value = st.number_input("", value=default, format="%.2f", key=feature)  # Users can change default values
    user_input.append(value)

if st.button('Predict'):
    # Convert user input to an array and reshape for the model
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    st.write(f'Predicted diabetes progression: {prediction[0]:.2f}')

