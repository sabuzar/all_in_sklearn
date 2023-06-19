from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
dataset=load_digits()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
X=df
y=dataset.target
scalaer=StandardScaler()
xscale=scalaer.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(xscale,y,test_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
pca=PCA(0.94)
xpca=pca.fit_transform(X)
X_trainpca,X_testpca,y_trainpca,y_testpca=train_test_split(xpca,y,test_size=0.3)
model.fit(X_trainpca,y_trainpca)
print(model.score(X_testpca,y_testpca))
