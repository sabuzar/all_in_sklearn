import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor
import seaborn as sns
from copy import deepcopy


data=pd.read_csv(r"C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\xclara.csv")
#print(data)

plt.rcParams['figure.figsize']=(16,9)
plt.style.use("ggplot")

f1=data['V1'].values
f2=data['V2'].values

X=np.array(list(zip(f1,f2)))

plt.scatter(f1,f2,c='b',s=7)
plt.show()
