import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
dataset=pd.read_csv('housing.csv')
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
dataset.dropna(inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True)
X = dataset.drop('Price', axis=1)
y = dataset['Price']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)
#todo l1
model=Lasso(alpha=50,max_iter=100,tol=0.1)
model.fit(train_X,train_y)
print(model.score(test_X,test_y))
#todo l2
model2=Ridge()
model2.fit(train_X,train_y)
print(model2.score(test_X,test_y))