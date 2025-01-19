import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
from get_data import get_data
from predict import predict

# Dummy options for stadiums and rounds
place_code = {'桐生':'01','戸田':'02','江戸川':'03','平和島':'04','多摩川':'05','浜名湖':'06','蒲郡':'07','常滑':'08','津':'09','三国':'10','びわこ':'11',"琵琶湖":"11", '住之江':'12','尼崎':'13','鳴門':'14','丸亀':'15','児島':'16','宮島':'17','徳山':'18','下関':'19','若松':'20','芦屋':'21','福岡':'22','唐津':'23','大村':'24'}
class_mapping = {'B2':1,'B1':2,'A2':3,'A1':4}
map_race = {'逃げ': 0, '差し': 1, 'まくり': 2, 'まくり差し': 3, '抜き':4, '恵まれ':5}
map_race = {'逃げ': 0, '差し': 1, 'まくり': 2, 'まくり差し': 3, '抜き':4, '恵まれ':5}
stadium_options = list(place_code.keys())
round_options = list(range(1, 13))
map_2ren = {12: 0}
l = 0
for i in range(1, 7):
    for j in range(1, 7):
        if i != j:
            map_2ren.update([(int(str(i) + str(j)), l)])
            l = l + 1

# 3連単
map_3ren = {123: 0}
l = 0
for i in range(1, 7):
    for j in range(1, 7):
        for k in range(1,7):
            if i != j and j != k and k != i:
                map_3ren.update([(int(str(i) + str(j) + str(k)), l)])
                l = l + 1

# Streamlit UI
st.title("Boat Racing Prediction AI")
st.sidebar.header("Select Parameters")

selected_date = st.sidebar.date_input("Select Date", value=date.today(), min_value=date.today() - timedelta(days=30), max_value=date.today() + timedelta(days=30))
selected_stadium = st.sidebar.selectbox("Select Stadium", stadium_options)
selected_round = st.sidebar.selectbox("Select Round", round_options)



def run_prediction(date, round, stadium):
    # Get data
    stadium = place_code[stadium]
    data = get_data(date, round, stadium)
    # Placeholder function to be implemented by the user
    matrix1, matrix2, matrix3 = predict(data)
    matrix1 = pd.DataFrame(matrix1, columns=list(map_2ren.keys())).T.sort_values(by=0, ascending=False).T
    matrix2 = pd.DataFrame(matrix2, columns=list(map_3ren.keys())).T.sort_values(by=0, ascending=False).T
    matrix3 = pd.DataFrame(matrix3, columns=list(map_race.keys())).T.sort_values(by=0, ascending=False).T
    return matrix1, matrix2, matrix3

st.write(f"### Selected Parameters")
st.write(f"- **Date:** {selected_date.strftime('%Y-%m-%d')}")
st.write(f"- **Stadium:** {selected_stadium}")
st.write(f"- **Round:** {selected_round}")

# Prediction button and result
if st.sidebar.button("Start Prediction"):
    matrix1, matrix2, matrix3 = run_prediction(selected_date, selected_round, selected_stadium)  # Call the function
    st.write("### Prediction Result")
    st.write("#### Matrix 1")
    st.dataframe(pd.DataFrame(matrix1))
    st.write("#### Matrix 2")
    st.dataframe(pd.DataFrame(matrix2))
    st.write("#### Matrix 3")
    st.dataframe(pd.DataFrame(matrix3))
else:
    st.write("### Prediction Result")
    st.write("(Press the 'Start Prediction' button to see the result)")
