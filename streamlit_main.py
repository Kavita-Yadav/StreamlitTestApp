import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

##loading the model from the saved file
# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

data = pd.read_csv("sales_by_segment.csv")

st.header("CLP Sales Prediction App")
st.text_input("Enter your Name: ", key="name")

if st.checkbox('Show dataframe'):
    data

st.subheader("Please select which data you want to predict!")
left_column, right_column = st.columns(2)
with left_column:
    inp_data_type = st.radio(
        'Data Types:',
        ('客戶分類 Customer Segment','售電量(百萬度) Sales(GWh)','客戶數量(千) Customer Number(000)'))


input_Length1 = st.slider('Day of Year', 1, 31, 2)
input_Length2 = st.slider('Day of Week', 0, 6, 1)
input_Length3 = st.slider('Quarter', 1, 4, 3)
input_Height = st.slider('Month', 1, 12, 3)
input_Width = st.slider('Year', 2023, 2026, 2023)


if st.button('Make Prediction') and inp_data_type == '售電量(百萬度) Sales(GWh)':
    inputs = np.expand_dims(
        [input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"The sales on selected date will be: {np.squeeze(prediction, -1):.2f} GWh")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
