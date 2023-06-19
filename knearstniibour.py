from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
dataset=pd.DataFrame(iris.data,columns=iris.feature_names)
print(dir(iris))
dataset['target']=iris.target
inputs=dataset.drop(['target'],axis='columns')
target=dataset.target
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3)
model=KNeighborsClassifier(n_neighbors=4)
model.fit(X_train,y_train)
print(round(model.score(X_test,y_test)*100),'%')
