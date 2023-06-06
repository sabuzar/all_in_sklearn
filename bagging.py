import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dataset=pd.read_csv('diiabetes.csv')
#print(dataset.isnull().sum())
#print(dataset.Outcome.value_counts())
inputs=dataset.drop('Outcome',axis='columns')
target=dataset.Outcome
scaler=StandardScaler()
scaled=scaler.fit_transform(inputs)
xtrain,xtest,ytrain,ytest=train_test_split(scaled,target,stratify=target,random_state=10)
kfold=cross_val_score(DecisionTreeClassifier(),scaled,target,cv=5)
print(kfold.mean())
print(0)
bag=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.9,
    oob_score=True,
    random_state=0
)
bag.fit(xtrain,ytrain)
print(bag.oob_score_)
scoreafterbag=cross_val_score(bag,scaled,target,cv=5)
print(scoreafterbag.mean())