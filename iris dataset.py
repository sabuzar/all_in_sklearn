import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(dataset.data,dataset.target,train_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
print(model.score(X_test,y_test))
y_predict=model.predict(X_test)
cm=confusion_matrix(y_test,y_predict)
plt.imshow(cm)
plt.title('iris flower type prediction')
plt.xlabel('petal len,petal width,sepal len,sepal width')
plt.ylabel('type of iris flower')
plt.tight_layout()
plt.show()