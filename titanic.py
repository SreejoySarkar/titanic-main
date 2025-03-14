import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('titanic_train.csv')
df.drop('Cabin',axis=1,inplace=True)
df['Age']=df['Age'].fillna(df['Age'].mode()[0])
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
sex_mapping={'male':1,'female':2}
df['Sex']=df['Sex'].map(sex_mapping)
embark_mapping={'S':1,'C':2,'Q':3}
df['Embarked']=df['Embarked'].map(embark_mapping)
survival_mapping={1:'survived',0:'Did not survived'}
df['Survival']=df['Survived'].map(survival_mapping)
df.drop(['PassengerId','Name','Ticket','Survived'],axis=1,inplace=True)
x=df.drop('Survival',axis=1)
y=df['Survival']
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
rfc_classifier = RandomForestClassifier()

rfc_classifier.fit(x_train, y_train)
pickle.dump(rfc_classifier,open('abc.pkl','wb'))
model=pickle.load(open('abc.pkl','rb'))

print("--------------")
