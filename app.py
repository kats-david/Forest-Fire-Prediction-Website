from lib2to3.pgen2.pgen import DFAState
from flask import Flask,request, url_for, redirect, render_template
import pickle
import sys
import os
import glob
import re
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from IPython.display import clear_output
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import time
import json


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image



app = Flask(__name__)
MODEL_PATH = 'my_model.h5'

model = load_model(MODEL_PATH)



@app.route('/')
def hello_world():
    return render_template("zaliwa.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    mldict = {
        "score" : 0,
        "risk_level": "low Risk",
        "risk_number": 0
    }
    inputs = [] 
    for x in request.form.values():
        inputs.append(x)
    # int_features=[float(x) for x in request.form.values()]
    # final=[np.array(int_features)]
    sample = [11.4, 32.6, 70.7, 33.0, 60, 80.9]
    scalar = MinMaxScaler()
    scalar.fit(sample)
    test_scaled = scalar.transform(sample)
    
    prediction=model.predict(test_scaled)

    confidence = max(prediction[0])
    risk_index = np.where(prediction[0] == confidence)
    print(inputs)
   
    # print(prediction[0].index(confidence))
    print(test_scaled)
    print(risk_index[0][0])
    if risk_index[0][0] == 0 : 
        mldict.update({"score":int (confidence*100), "risk_level": "low Risk", "risk_number": int (risk_index[0][0]) })
        # return mldict
    elif risk_index[0][0] == 1:
        mldict.update({"score":int (confidence*100), "risk_level": "Medium Risk", "risk_number": int (risk_index[0][0] )})
    elif risk_index[0][0] == 2:
        mldict.update({"score":int (confidence*100), "risk_level": "High Risk", "risk_number": int (risk_index[0][0]) })
    else:
        print("enter valid input")
    
    print(mldict)
    data = json.dumps(mldict)
    return data


if __name__ == '__main__':
    app.run(debug=True)
