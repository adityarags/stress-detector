import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import re

model = pickle.load(open("rf_model.h5", "rb"))

st.title("Stress Detection")

sentence = st.text_input('Describe your mental state in 50 words')
data = pd.read_csv("dataset.csv")
print(data.head())
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english'))


def clean(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = [word for word in text.split(' ') if word not in stopword]
  text = " ".join(text)
  text = [stemmer.stem(word) for word in text.split(' ')]
  text = " ".join(text)
  return text


data["text"] = data["text"].apply(clean)

x = np.array(data["text"])

cv = CountVectorizer()
X = cv.fit_transform(x)
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english'))


def clean(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = [word for word in text.split(' ') if word not in stopword]
  text = " ".join(text)
  text = [stemmer.stem(word) for word in text.split(' ')]
  text = " ".join(text)
  return text


data["text"] = data["text"].apply(clean)

data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

data = cv.transform([sentence]).toarray()
submit = st.button("Submit")
if submit:
  param = data

  op = model.predict(param)
  if op == "Stress":
    st.warning('Stress Detected!', icon="⚠️")
  else:
    st.success('Stress not detected!', icon="✨")
