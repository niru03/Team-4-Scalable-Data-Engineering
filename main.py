# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\niran's\UNH\projects\team 4 Scalable Data Engineering\hotel_bookings.csv")

# Data preprocessing
def preprocess_data(df):
    # Handle missing values
    df.fillna(0, inplace=True)
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

# Load and preprocess data
data = load_data()
data = preprocess_data(data)

# Model training
X = data.drop('is_canceled', axis=1)
y = data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Streamlit app
st.title("Hotel Booking Analysis and Cancellation Prediction")
st.markdown('This application is a Streamlit dashboard that can be used '
            'to predict hotel booking cancellations.')

# Show the dataframe
if st.checkbox('Show dataframe'):
    st.write(data)

# From where are the most guests coming?
st.subheader('From where are the most guests coming?')
country_count = data['country'].value_counts().rename_axis('country').reset_index(name='counts')
st.bar_chart(country_count)

# How much do guests pay for a room per night?
st.subheader('How much do guests pay for a room per night?')
st.bar_chart(data['adr'])

# Select a month to see the price variation over the year and the demand
month = st.selectbox('Select a month to see the price variation over the year and the demand:', data['arrival_date_month'].unique())
st.subheader(f'How does the price per night vary in {month}?')
st.line_chart(data[data['arrival_date_month']==month]['adr'])

# Demand in selected month
st.subheader(f'Demand in {month}')
st.bar_chart(data['arrival_date_month'].value_counts())

# How long do people stay at the hotels?
st.subheader('How long do people stay at the hotels?')
# Check if 'total_nights' column exists
if 'total_nights' in data.columns:
    st.bar_chart(data['total_nights'])
else:
    st.write("The column 'total_nights' does not exist in the data.")

# Predict if a booking will be canceled
if st.button('Predict if a booking will be canceled'):
    user_input = st.text_input("Enter the booking details:")
    prediction = model.predict(user_input)
    st.subheader('Prediction')
    st.write(prediction)
