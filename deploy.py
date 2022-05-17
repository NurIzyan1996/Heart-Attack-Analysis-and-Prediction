# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:03:35 2022
Objection: To deploy the saved model to predict new outcomes
@author: Nur Izyan Binti Kamardin
"""
import os
import pickle
import numpy as np
import streamlit as st

## paths
MMS_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'best_model.pkl')

# ML model loading
mms_scaler = pickle.load(open(MMS_PATH,'rb'))
model = pickle.load(open(MODEL_PATH,'rb'))

with st.form('DHeart Attack Prediction Form'):
    st.write("Patient's info")
    age = int(st.number_input('Age of the patient'))
    sex = int(st.number_input('Sex of the patient ~ 0 = Female, 1 = Male '))
    cp = int(st.number_input('Chest pain type ~ 0 = Typical Angina, \
                             1 = Atypical Angina, 2 = Non-anginal Pain, \
                                 3 = Asymptomatic'))
    trtbps = int(st.number_input('Resting blood pressure (in mm Hg)'))
    chol = int(st.number_input('Cholestoral in mg/dl fetched via BMI sensor'))
    fbs = int(st.number_input('fasting blood sugar > 120 mg/dl ~ 1 = True, \
                              0 = False'))
    restecg = int(st.number_input('Resting electrocardiographic results ~ \
                                  0 = Normal, 1 = ST-T wave normality, \
                                      2 = Left ventricular hypertrophy'))
    thalachh = int(st.number_input('Maximum heart rate achieved'))
    exng = int(st.number_input('Exercise induced angina ~ 1 = Yes ~ 0 = No'))
    oldpeak = st.number_input('Previous peak')
    slp = int(st.number_input('Slope'))
    caa = int(st.number_input('Number of major vessels'))
    thall = int(st.number_input('Thalium Stress Test result \
                                ~ Range from 0 to 3'))
    
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patient_info = np.array([age, sex, cp, trtbps, chol, fbs, restecg, 
                                 thalachh, exng, oldpeak, slp, caa, thall])
        
        patient_info = mms_scaler.transform(np.expand_dims(patient_info, 
                                                           axis=0))
        
        outcome = model.predict(patient_info)
        
        st.write(np.argmax(outcome))
        
        if np.argmax(outcome)==1:
            st.warning('High chance of heart attack, GOOD LUCK')
        else: 
            st.balloons()
            st.success('YEAH! Less chance of heart attack')

