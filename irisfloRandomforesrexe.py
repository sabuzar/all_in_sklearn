from sklearn.datasets import  load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
x=df.drop(['target'],axis='columns')
y=df.target
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=RandomForestClassifier(n_estimators=5)
model.fit(X_train,y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))
