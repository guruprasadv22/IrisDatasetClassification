import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title("""Flower Class Prediction using Iris dataset""")
st.text('This app uses random forest classifier to predict the Iris Flower type')

def slider_input():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.1)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 4.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.1)

    dict1 = {'Sepal_length': sepal_length, 'Sepal_width': sepal_width,
             'Petal_length': petal_length, 'Petal_width': petal_width}

    features = pd.DataFrame(dict1, index = [0])

    return features


df = slider_input()

iris = datasets.load_iris()

X = iris.data
y = iris.target

st.subheader("Labels in the classification problem with their index")
st.write(iris.target_names)

model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(df)
st.subheader("The flower type is predicted to be: ")
st.write(iris.target_names[prediction])

prob = model.predict_proba(df)
st.subheader("Prediction probabilites: ")
st.write(prob)

