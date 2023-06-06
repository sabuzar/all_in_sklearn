import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
data=pd.read_csv(r'C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\company.csv')
print(data)
independentvar=data.drop(['salary_more_then_100k'],axis='columns')
targetvar=data['salary_more_then_100k']

le_comp=LabelEncoder()
#le_job=LabelEncoder()
#le_deg=LabelEncoder()

independentvar['comp_n']=le_comp.fit_transform(independentvar['company'])
independentvar['job_n']=le_comp.fit_transform(independentvar['job'])
independentvar['deg_n']=le_comp.fit_transform(independentvar['degree'])

data2=independentvar.drop(['company','job','degree'],axis='columns')

X_train,X_test,y_train,y_test=train_test_split(data2,targetvar,test_size=0.3)

model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.predict([[2,0,1]]))
print(model.score(X_test,y_test))

