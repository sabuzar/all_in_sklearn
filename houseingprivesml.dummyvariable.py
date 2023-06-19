import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv(r'C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\town1.csv')
#todo first way
dummyvariable=pd.get_dummies(data.town)
mergingtwodataframes=pd.concat([data,dummyvariable],axis='columns')
finaldatframe=mergingtwodataframes.drop(['town','west windsor'],axis='columns')
x=finaldatframe.drop(['price'],axis='columns')
y=finaldatframe.drop(['area','monroe township','robinsville'],axis='columns')
model=LinearRegression()
model.fit(x,y)
print(model.predict([[2800,0,1]]))
print(model.predict([[3400,0,0]]))
print(model.score(x,y)*100)

