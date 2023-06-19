import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
x=np.linspace(1,100,100)
y=x*2
#print(x)
y[10:30]=np.random.rand(20)*120+100
x=x.reshape(-1,1)
y=y.reshape(-1,1)
#print(y)
#todo data visualization
def plot(clf=None,clf_name="",color=None):
    plt.scatter(x,y,label="sample")
    plt.title("outleter")
    if clf is not None:
        y_predit=clf.predict(x)
        plt.plot(x,y_predit,label="clf name",color=color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot()
lr=LinearRegression().fit(x,y)
plot(lr,color="k")
tr=TheilSenRegressor().fit(x,y)
plot(lr,'ols',color="b")