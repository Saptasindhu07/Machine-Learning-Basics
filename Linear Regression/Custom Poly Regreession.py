import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

degree=int(input("Enter the Degree: "))
copy=degree
X = 6 * np.random.rand(200)
y=0
l=[]
for i in range(degree+1):
    l.append(X**copy)
    copy=copy-1
converted=np.array(l)

coef_real=80*np.random.rand(degree+1)

for i in range(degree+1):
    y+=coef_real[i]*converted[i]+2000*np.random.rand(200)

_l_=[]
for j in range(len(X)):
    copy_=degree
    k=[]
    for i in range(degree+1):
        k.append(X[j]**copy_)
        copy_=copy_-1
    _l_.append(k)
converted2=np.array(_l_)
    
print(converted2.shape)

lr2=LinearRegression()
lr2.fit(converted2,y)

print(lr2.predict(converted2))

plt.scatter(X,lr2.predict(converted2),color='r')
plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()