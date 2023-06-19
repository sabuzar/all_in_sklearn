from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
data=load_iris()
df=pd.DataFrame(data.data,columns=data['feature_names'])
df['flower']=data.target
df['flower']=df['flower'].apply(lambda x: data.target_names[x])
inputs=df.drop(['flower'],axis='columns')
target=df.flower
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3)
model_parameters={
    'svm':{
        'model':SVC(gamma='auto'),
        'parameters':{
            'C':[1,10,20],
            'kernel':['linear','rbf']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'parameters':{
            'n_estimators':[1,5,10]
        }
    },
    'logisticregression':{
        'model':LogisticRegression(solver='liblinear',multi_class='auto'),
        'parameters':{
            'C':[1,5,10]
        }
    }
}
scores=[]
for model_name,mp in model_parameters.items():
    clf=GridSearchCV(mp['model'],mp['parameters'],cv=5,return_train_score=False)
    clf.fit(inputs,target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_parameters':clf.best_params_
    })
df2=pd.DataFrame(scores,columns=['model','best_score','best_parameters'])
df2.to_csv('hyperperameter1.csv')











