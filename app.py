import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv('disease_data.csv')
X=data.drop('disease',axis=1)
y=data['disease']
model=DecisionTreeClassifier()
model.fit(X,y)
st.title("Disease Prediction Based on Symptoms")
st.write("select your symptoms and click **Predict** to see the possible disease.")
fever = st.checkbox("Fever 🤒")
cough = st.checkbox("Cough 😷")
headache = st.checkbox("Headache 🤕")
body_pain = st.checkbox("Body Pain 💢")
fatigue = st.checkbox("Fatigue 😴")

# Convert checkbox to numeric input (1 = yes, 0 = no)
symptoms_input = [
    int(fever),
    int(cough),
    int(headache),
    int(body_pain),
    int(fatigue)
]

# Predict when button is clicked
if st.button("Predict"):
    prediction = model.predict([symptoms_input])
    st.success(f"🧠 You might have: **{prediction[0]}**")