from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
data=load_iris()
df=pd.DataFrame(data.data,columns=data['feature_names'])
df['flower']=data.target
df['flower']=df['flower'].apply(lambda x: data.target_names[x])
inputs=df.drop(['flower'],axis='columns')
target=df.flower
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)
model=SVC()
model.fit(X_train,y_train)
clf=GridSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)
clf.fit(X_train,y_train)
df2=pd.DataFrame(clf.cv_results_)
df2.to_csv('df2.csv')

clf2=GridSearchCV(RandomForestClassifier(max_features='auto'),{
    'n_estimators':[10,19,20,100],
},cv=5,return_train_score=False)
clf2.fit(X_train,y_train)
df4=pd.DataFrame(clf2.cv_results_)
df4.to_csv('df4.csv')














