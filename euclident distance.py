import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
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
#plt.show()

def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)
k=3
C_x=np.random.randint(0,np.max(X)-20,size=k)
C_y=np.random.randint(0,np.max(X)-20,size=k)

C=np.array(list(zip(C_x,C_y)),dtype=np.float32)

C_old=np.zeros(C.shape)
clusters=np.zeros(len(X))
#todo error fun
error=dist(C,C_old,None)
while error!=0:
    for i in range(len(X)):
        distances=dist(X[i],C)
        cluster=np.argmin(distances)
        clusters[i]=cluster
    C_old=deepcopy(C)
    for i in range(k):
        points=[X[j] for j in range(len(X))if clusters[j]==i]
        C[i]=np.mean(points,axis=0)
    error=dist(C,C_old,None)
colors=['b', 'g', 'r', 'c', 'm', 'y']
fig,ax=plt.subplots()
for i in range(k):
    points=np.array([X[j] for j in range(len(X))if clusters[j]==i])
    ax.scatter(points[:,0],points[:,1],s=7,c=colors[i])
ax.scatter(C[:,0],C[:,1],marker='*',s=200,c='k')
plt.show()

