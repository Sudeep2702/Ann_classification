import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import pickle

loaded_model = tf.keras.models.load_model("ann_model_new.h5")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('hot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=
50000.0)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

from sklearn.preprocessing import LabelEncoder
gender = label_encoder.transform([gender])
hot_encoded_data = onehot_encoder.transform([[geography]])
get_encoded_df = pd.DataFrame(hot_encoded_data, columns = onehot_encoder.get_feature_names_out(['Geography']))

input_data = pd.DataFrame({})

numeric_data = pd.DataFrame([[
    credit_score, gender, age, tenure, balance, num_of_products,
    has_cr_card, is_active_member, estimated_salary
]], columns=[
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
])
# ---- Combine both numeric and encoded data ----
final_input = pd.concat([numeric_data, get_encoded_df], axis=1)

# ---- Scale data ----
input_scaled = scaler.transform(final_input)

# ---- Prediction Button ----
if st.button("Predict"):
    prediction = loaded_model.predict(input_scaled)
    if prediction[0][0] > 0.5:
        st.error("🚨 The customer is likely to stay.")
    else:
        st.success("✅ The customer is likely to exit.")

    st.write("Model Output:", float(prediction[0][0]))

