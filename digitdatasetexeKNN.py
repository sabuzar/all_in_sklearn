from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data=load_digits()

print(dir(data))
print(data.data[0])

print(data.feature_names)
print()
X_train,X_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2)
module=KNeighborsClassifier(n_neighbors=5)
module.fit(X_train,y_train)
print(module.score(X_test,y_test))
print(module.predict([data.data[0]]))
print(data.target[0])