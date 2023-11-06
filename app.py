import numpy as np
from flask import Flask, request, render_template
import joblib
import pickle
import string
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import wordcloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 




app = Flask(__name__)
# model = joblib.load(open('spam_sgd.pkl', 'rb'))
# cv = joblib.load(open('spam_tfvector.pkl', 'rb'))

def remove_punctuation_and_stopwords(sms):
    
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    
    sms_no_punctuation_no_stopwords = \
        [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
        
    return sms_no_punctuation_no_stopwords



model = open('spam_sgd.pkl','rb')
model = pickle.load(model)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        text = request.form['Review']
        prediction = model.predict([text])
        data = [text]
        # vectorizer = cv.transform(data).toarray()
        # prediction = model.predict(vectorizer)
        print(prediction)


    if prediction[0]:
        return render_template('index.html', prediction_text='The Message is SPAM')
    else:
        return render_template('index.html', prediction_text='The Message is not SPAM')


if __name__ == "__main__":
    app.run(debug=True)
