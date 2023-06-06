import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
dataset=pd.read_csv('eart.csv')
inputs=dataset.drop(['ChestPainType','RestingECG','ST_Slope','HeartDisease'],axis='columns')
target=dataset.HeartDisease

inputs['Sex']=dataset['Sex'].apply(lambda x:1 if x=='M' else 0)
inputs['ExerciseAngina']=dataset['ExerciseAngina'].apply(lambda z:1 if z=='Y' else 0)
#inputs['ST_Slope']=dataset['ST_Slope'].apply(lambda y:1 if y=='Flat' else 0)
xtrain,xtest,ytrain,ytest=train_test_split(inputs,target,test_size=0.2,random_state=10)
baged=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,
    max_samples=0.8,
    random_state=0
)
baged.fit(xtrain,ytrain)
print(baged.oob_score_)
clone1=cross_val_score(DecisionTreeClassifier(),inputs,target,cv=5)
print(clone1.mean())
clone2=cross_val_score(baged,inputs,target,cv=5)
print(clone2.mean())

baged2=BaggingClassifier(
    base_estimator=SVC(),
    n_estimators=100,
    oob_score=True,
    max_samples=0.8,
    random_state=0
)
baged2.fit(xtrain,ytrain)
print(baged2.oob_score_)
clone3=cross_val_score(baged2,inputs,target,cv=5)
print(clone3.mean())