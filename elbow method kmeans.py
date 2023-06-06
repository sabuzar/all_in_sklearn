import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv(r"C:\Users\Asad Laptop\Downloads\Projects and Case Studies on Machine Learning\Name.csv")


#todo MinMaxScaler()
scaler=MinMaxScaler()
scaler.fit(data[['Income($)']])
data['Income($)']=scaler.transform(data[['Income($)']])

scaler.fit(data[['Age']])
data['Age']=scaler.transform(data[['Age']])






#todo kmeans cluster
cluster=KMeans(n_clusters=3)
ypredict=cluster.fit_predict(data[['Age','Income($)']])
print(ypredict) 
data['clusters']=ypredict

#todo visualize
df0=data[data.clusters==0]
df1=data[data.clusters==1]
df2=data[data.clusters==2]
centroid=cluster.cluster_centers_

plt.scatter(df0.Age,df0['Income($)'])
plt.scatter(df1.Age,df1['Income($)'])
plt.scatter(df2.Age,df2['Income($)'])
plt.scatter(centroid[:,0],centroid[:,1],marker='*',c='k')
plt.show()


elbow=range(1,10)
sse=[]
for k in elbow:
    km=KMeans(n_clusters=k)
    km.fit(data[['Age','Income($)']])
    sse.append(km.inertia_)
plt.xlabel('k')
plt.ylabel('sse')
plt.plot(elbow,sse)
plt.show()