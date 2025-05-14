import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

## Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load the encoder and scaler
with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App
st.title("Customer Churn Prediction")

## User Input

# 1. Geography (Dropdown)
geography = st.selectbox("Geography", onehotencoder.categories_[0])

# 2. Gender (Dropdown)
gender = st.selectbox("Gender", label_encoder_gender.classes_)

# 3. Age (Slider)
age = st.slider("Age", 18, 92)

# 4. Credit Score (Slider)
credit_score = st.number_input("Credit Score")

# 5. Balance (Number input)
balance = st.number_input("Balance")

# 6. Estimated Salary
estimated_salary = st.number_input("Estimated Salary")

# 7. Tenure (Slider)
tenure = st.slider("Tenure (Years with Bank)", 0, 10)

# 8. Number of Products (Dropdown)
num_of_products = st.slider("Number of Products", 1, 4)

# 9. Has Credit Card (Yes/No to 1/0)
has_cr_card = st.selectbox("Has Credit Card?", [0,1])
#has_cr_card = 1 if has_cr_card == "Yes" else 0

# 10. Is Active Member (Yes/No to 1/0)
is_active_member = st.selectbox("Is Active Member?", [0,1])
#is_active_member = 1 if is_active_member == "Yes" else 0

# Button to predict (assuming model is loaded)
if st.button("Predict"):
    # Display user inputs
    st.write("User Inputs:")
    input_data = ({
        "Geography": [geography],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Credit Score": [credit_score],
        "Balance": [balance],
        "Estimated Salary": [estimated_salary],
        "Tenure": [tenure],
        "Num of Products": [num_of_products],
        "Has Credit Card": [has_cr_card],
        "Is Active Member": [is_active_member]
    })

    st.write(input_data)

    ## Onehot encode Geography
    geo_encoded = onehotencoder.transform([[geography]]).toarray()
    geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

    ## Create DataFrame from user input data
    user_input_df = pd.DataFrame([input_data])

    ## Combine user input and one-hot encoded geography data
    input_data_combined = pd.concat([user_input_df.reset_index(drop=True), geo_encoder_df], axis=1)

    ## Scale the data
    input_data_scaled = scaler.transform(input_data_combined)

    ## Predict the data
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
