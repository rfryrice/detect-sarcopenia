import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

data = pd.read_excel('sarc_data.xlsx')
fem_model = pk.load(open('femodel.pkl', 'rb'))
male_model = pk.load(open('mamodel.pkl', 'rb'))


def process(model_vars):
    a = np.array()
    for i,v in enumerate(model_vars):
        print(v)
        #np.insert(a,i,v)
    return a

st.title('Detecting Sarcopenia')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 40, 100)
    weight = st.number_input("Weight (kg)", 40, 100)
    height = st.number_input("Height (cm)", 130, 200)
    waist = st.number_input("Waist (cm)", 50, 200)
    hip = st.number_input("Hip (cm)", 10, 200)

with col2:
    alcohol = st.checkbox("Alcohol")
    smoker = st.checkbox("Smoker")
    if smoker:
        smoke_packets = st.number_input("Packets per year", 1, 200)
    dm = st.checkbox("Diabetes Mellitus")
    if dm:
        insulin = st.checkbox("Insulin")
    ht = st.checkbox("Hypertension")
    hthyroid = st.checkbox("Hypothyroidism")
    hlipid = st.checkbox("Hyperlipidemia")

with col3:
    exercise = st.selectbox("How often do you exercise?",
                            ["0", "1-2 times/week", "3-4 times/week"])
    work_stat = st.selectbox("What is your working status?",
                             ["none", "retired", "unemployed", "full-time/part-time work",
                              "working", "housewife", "full-time work", "part-time work",
                              "not working", "abstinence"])
    cst = st.number_input("Chair Stand Test", 0, 60)
    gspd = st.number_input("Gait Speed", 0., 2.5)
    grs = st.number_input("Grip Strength", 0, 80)
    gender = st.selectbox('Gender', ('Male', 'Female'))
    st.write('You selected:', gender)

predict = st.button("Run Predict")

model_vars = [
    age, weight, height, hip, ht, dm, insulin, hthyroid,
    hlipid, smoker, smoke_packets, exercise, alcohol,
    work_stat, cst, gspd, grs
]

if predict:
    estimators = process(model_vars)
    if gender == 'Male':
        male_model.predict(estimators)
    elif gender == 'Female':
        fem_model.predict(estimators)
