import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
data=pd.read_csv(r'C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\Car Model.csv')
#print(data)
makingdummy=pd.get_dummies(data['Car Model'])
#print(makingdummy)
concat=pd.concat([data,makingdummy],axis='columns')
#print(concat)
finalcsvdata=concat.drop(['Car Model','BMW X5'],axis='columns')
#print(finalcsvdata)
x=finalcsvdata.drop(['Sell Price($)'],axis='columns')
#print(x)
y=finalcsvdata['Sell Price($)']
#print(y)
model=LinearRegression()
model.fit(x,y)
print(model.predict([[45000,4,0,1]]))
#print(model.score(x,y)*100)
#todo savi model
with open('model_carprice','wb') as f:
    pickle.dump(model,f)
with open('model_carprice','rb') as f:
    md=pickle.load(f)
print(md.predict([[45000,4,0,1]]))