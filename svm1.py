from sklearn.datasets import load_iris
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df['targetnames']=df.target.apply(lambda x:data.target_names[x])
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'])
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'])
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'])
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'])
plt.show()
X=df.drop(['target','targetnames'],axis='columns')
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=SVC(C=1)
print(model.fit(X_train,y_train))
print(model.predict(X_test))
print(model.score(X_test,y_test))
