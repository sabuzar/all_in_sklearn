import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#todo idea of backward elimination is to iloc data with negative indexes

data=pd.read_csv(r"C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\50_Startups.csv")
x=data.iloc[:,-2]
x1=data.iloc[:,-1]
y=data['Profit']

#todo dummies
#x=pd.get_dummies(x,columns=data['State'])
#print(x.head(6))

#todo data visualzation
plt.scatter(x,y)
plt.title('backwardelimination')
plt.xlabel("amount spend")
plt.ylabel("profit")
plt.show()