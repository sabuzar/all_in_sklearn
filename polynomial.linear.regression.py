import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv(r'C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\Position_Salaries.csv')
x1=data.iloc[:,1]
x=data.iloc[:,1:2]
y=data.iloc[:,2]
#print(x,x1,data,y)
rg=LinearRegression()
rg.fit(x,y)
#todo ploynomial regression
poly_reg2=PolynomialFeatures(degree=2)
poly_reg3=PolynomialFeatures(degree=3)
x_poly2=poly_reg2.fit_transform(x)
x_poly3=poly_reg3.fit_transform(x)
len_reg_poly2=LinearRegression().fit(x_poly2,y)
len_reg_poly3=LinearRegression().fit(x_poly3,y)
#print(len_reg_poly2.predict(poly_reg2.fit_transform(x)))
#todo data visulatizatiion
plt.scatter(x,y,color="k")
plt.plot(x,rg.predict(x),color="r")
plt.plot(x,len_reg_poly2.predict(poly_reg2.fit_transform(x)),color="g")
plt.plot(x,len_reg_poly3.predict(poly_reg3.fit_transform(x)),color="k")
plt.tight_layout()
plt.show()