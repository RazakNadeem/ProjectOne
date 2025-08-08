import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("disease_data.csv")
X=data.drop('disease',axis=1)
y=data['disease']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
print("Enter your Symptoms(1 for Yes,0 for No):")
fever=int(input("Do you have fever?"))
cough=int(input("Do you have cough?"))
headache=int(input("Do you have headache?"))
body_pain=int(input("Do you have body pain?"))
fatigue=int(input("Do you feel fatigue?"))
user_input=[[fever,cough,headache,body_pain,fatigue]]
prediction=model.predict(user_input)
print(f"\n You might have:{prediction[0]}")