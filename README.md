# Heart Attack Analysis and Prediction
 Creating Machine Learning model to predict patients' heart condition.

# Description

This repository contains 2 python files (train.py and deploy.py).

train.py contains the codes to extract the best machine learning model with the highest accuracy and train on Heart.csv dataset.

deploy.py contains the codes to deploy the machine learning model and launch the streamlit app.

The user can also use the Streamlit app to insert their health information.

# How to use

1. Clone this repository and use the best_model.pkl and mms_scaler.pkl (inside saved_model folder) to deploy on your dataset.
2. Another option to predict new data is by inserting input in the Streamlit app.
3. Run streamlit via conda prompt by activating the correct environment and working directory and run the code "streamlit run deploy.py".
4. Your browser will automatically redirected to streamlit local host and the streamlit app can now be launched.
5. Insert your health information and click "Submit" button to view the result.

# Performance of the model
The photo shows the accuracy score, classification report and confusion matrix of the machine leraning model.
![Performance of the model](model_performances.JPG)


# Streamlit Image from my browser
![Streamlit Image from my browser](streamlit_app.JPG)

# Credit

Shout out to the owner of the heart dataset https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
