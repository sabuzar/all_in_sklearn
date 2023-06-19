import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
data=pd.read_csv('pd.csv')
data=data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
target=data.Survived
inputs=data.drop(['Survived'],axis='columns')
inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
dns1=math.floor(inputs.Age.median())
inputs.Age=inputs.Age.fillna(dns1)
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3 )
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
with open('model1.2','wb') as f:
    pickle.dump(model,f)
with open('model1.2','rb') as f:
    mp=pickle.load(f)
print(mp.predict(X_test[:10]))