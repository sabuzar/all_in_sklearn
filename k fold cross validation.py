import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
digit=load_digits()
df=pd.DataFrame(digit.data)
df['target']=digit.target
x=df.drop(['target'],axis='columns')
y=df.target
print(cross_val_score(LogisticRegression(),x,y,cv=10))
print(cross_val_score(SVC(),x,y,cv=10))
print(cross_val_score(RandomForestClassifier(),x,y,cv=10))