import streamlit as st
import pandas as pd
import numpy as np
import joblib

data = pd.read_excel('sarc_data.xlsx')
data = data.drop(data.columns[0], axis=1)
fem_model = joblib.load('femodel.pkl')
male_model = joblib.load('mamodel.pkl')

st.set_page_config(
    page_title="Detecting Sarcopenia",
    page_icon=':random:',
    layout='centered'
)

with st.sidebar:
    st.markdown("""### Background:
Age-related loss of skeletal muscle mass and function, defined as sarcopenia, is significantly
related to adverse health outcomes e.g. increased risk of falls and fractures, physical frailty,
mobility limitation, and even premature mortality. Early detection of risk of disease allows hospitals to
free up resources and for patients to dismiss extra tests.

### Modeling:
Gradient Boosting Classifier via `scikit-learn` was trained on the dataset shown. Of 30+ features, 19 are used.
Two models were trained based on gender due to sample imbalance. The Female:Male sample ratio is 3:1

Male F1-score: 0.74
Female F1_score: 0.54

Male model accuracy is 89.47% 
Female model accuracy is 85.71%

Cross-validation: 5

**DISCLAIMER:** This app is not intended to diagnose disease. Please consult a medical professional
if you believe you may be at risk of Sarcopenia.
""")

st.title('Detecting Sarcopenia')
st.header("Dataframe")
st.dataframe(data)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 40, 100)
    weight = st.number_input("Weight (kg)", 40., 100.)
    height = st.number_input("Height (cm)", 130., 200.)
    waist = st.number_input("Waist (cm)", 50., 200.)
    hip = st.number_input("Hip (cm)", 10., 200.)

with col2:
    alcohol = st.selectbox("Alcoholic, if so specify",
                           ["0", "social", "regular"])
    smoker = st.checkbox("Smoker")
    smoke_packets = st.number_input("Packets per year", 0, 200)
    dm = st.checkbox("Diabetes Mellitus")
    insulin = st.checkbox("Insulin")
    ht = st.checkbox("Hypertension")
    hthyroid = st.checkbox("Hypothyroidism")
    hlipid = st.checkbox("Hyperlipidemia")

with col3:
    exercise = st.selectbox("How many times do you exercise?",
                            ["0", "1-2/week", "3-4/week"])
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


def gen_model():
    model_label = [
        'Age', 'Weight', 'Height', 'Hip', 'HT', 'DM', 'Insulin', 'Hypothyroidism',
        'Hyperlipidemia', 'Smoking', 'Smoking (packet/year)', 'Exercise', 'Alcohol',
        'Working Status', 'CST', 'Gait speed', 'Grip strength'
    ]
    model_vars = [
        age, weight, height, hip, ht, dm, insulin, hthyroid,
        hlipid, smoker, smoke_packets, exercise, alcohol,
        work_stat, cst, gspd, grs
    ]

    model = pd.DataFrame([model_vars], columns=model_label)

    return model


if predict:
    model = gen_model()
    if gender == 'Male':
        proba = male_model.predict_proba(model)
        p = male_model.predict(model)
    elif gender == 'Female':
        proba = fem_model.predict_proba(model)
        p = fem_model.predict(model)
    print(p)

    if p:
        st.info(
            f"You have been flagged for sarcopenia \n with probability of {proba[0,1]:.3f}")
    else:
        st.info("You likely do not have sarcopenia")
        st.balloons()
