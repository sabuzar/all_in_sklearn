import math
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
dataimp=pd.read_csv('pnj1.csv')
input=dataimp.drop(['Survived','PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')

input.Sex=input.Sex.map({'male':1,'female':2})
input.Sex=input.Sex.fillna(1)
input.Age=input.Age.fillna(input.Age.mean())
print(input.head(10))
target=dataimp.Survived
print(target.head(4))
X_train,X_test,y_train,y_test=train_test_split(input,target,test_size=0.2)
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))