import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv(r'C:\Users\Asad Laptop\Desktop\ml\HR_comma_sep.csv')
subdata=data[['satisfaction_level','average_montly_hours','left','promotion_last_5years','salary']]
creatingdumies=pd.get_dummies(subdata.salary,prefix='salary')
finalcsvdata=pd.concat([subdata,creatingdumies],axis='columns')
X=finalcsvdata.drop(['salary','left'],axis='columns')
y=subdata['left']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
print(model.predict(X_test))
print(model.score(X_test,y_test))

