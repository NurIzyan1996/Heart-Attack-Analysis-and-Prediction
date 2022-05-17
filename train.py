# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:47:48 2022
Objective: Extracting the best Machine Learning model 
            to predict the chance of a person having a heart attack.
@author: Nur Izyan Binti Kamarudin
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


#%% Paths
HEART_PATH = os.path.join(os.getcwd(), 'heart.csv')
MMS_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'best_model.pkl')

#%%
# STEP 1: Data Loading
df = pd.read_csv(HEART_PATH)

#%%
# STEP 2: Data Inspection
# a) display the first 10 rows of data
print(df.head(10))

# b) view the summary, non-null
print(df.info())
# Observation: there is no null value

# c) view the statistics of data
print(df.describe().T)

# d) visualize data distributions
df.boxplot()

#%%
# STEP 3: Data Cleaning
# a) remove duplicate
df = df.drop_duplicates()

#%%
# STEP 4: Data Preprocessing

# a) extract features(X) and target(Y)
X = df.drop(['output'],axis=1)
y = df[['output']]

# b) Standardize the features using MinMaxScaler approach (result:positive scale)
# the categorical data is already encoded
mms_scaler = MinMaxScaler()
X = mms_scaler.fit_transform(X)
pickle.dump(mms_scaler, open(MMS_SAVE_PATH, 'wb'))

#%%
# STEP 6: ML Pipeline Model
# a) split train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)

# b) create steps 
steps_knc = [('KNeighbors Classifier', KNeighborsClassifier())]
steps_tree = [('Decision Tree Classifier', DecisionTreeClassifier())]
steps_forest = [('Random Forest Classifier', RandomForestClassifier())]
steps_log = [('Logistic Regression', LogisticRegression(solver='liblinear'))]
steps_svm = [('SVM', SVC())]

# c) load the steps into the pipeline 
pipeline_knc = Pipeline(steps_knc)
pipeline_tree = Pipeline(steps_tree)
pipeline_forest = Pipeline(steps_forest)
pipeline_log = Pipeline(steps_log)
pipeline_svm = Pipeline(steps_svm)

# d) create a list to store all pipelines
pipelines = [pipeline_knc, pipeline_tree, pipeline_forest, pipeline_log, 
             pipeline_svm] 

# e) fit the training data into the pipeline
for pipe in pipelines:
    pipe.fit(X_train,y_train)

# f) extract the accuracy score of these ML models
pipe_dict = {0:'KNeighbors Classifier', 1:'Decision Tree Classifier', 
             2:'Random Forest Classifier', 3:'SVM', 4:'Logistic Regression'}

predictions = [] # prediction of all models
best_score = 0
best_scaler = 0
best_pipeline = ''

for i,model in enumerate(pipelines):
    predictions.append(model.predict(X_test))
    print("{} Test Accuracy:{}".format(pipe_dict[i], 
                                       model.score(X_test,y_test)))
    if model.score(X_test, y_test) > best_score:
        best_score = model.score(X_test, y_test)
        best_scaler = i        
        best_pipeline = model
        
print('Best Model is {} with accuracy of \
      {}%'.format(pipe_dict[best_scaler],(best_score)*100))
# Observation: Random Forest Classifier produces the highest accuracy score.

best_pipeline_prediction = predictions[2]
print(classification_report(y_test, best_pipeline_prediction))
print(confusion_matrix(y_test, best_pipeline_prediction))

# h) save the prediction ML model
with open(MODEL_SAVE_PATH, 'wb') as file:
    pickle.dump(best_pipeline, file)

