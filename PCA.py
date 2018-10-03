#牛叉叉
import numpy as np 
import matplotlib.pyplot as plt 
x = np.linspace(1, 10, 50)
y = 2*x+1+5*np.random.rand(50)
plt.scatter(x,y,c = "b")

def pca(X, feature_num):
    meanVals = np.mean(X, axis = 1)
    meanRemoved = X-meanVals.reshape((2,1))
    X1 = np.dot(X, X.T)
    eigVals, eigVects = np.linalg.eig(X1)
    print(eigVals,eigVects)
    eigVals_index = np.argsort(eigVals)[::-1]
    ##eigVals_select = eigVals[:-1*feature_num-1:-1]
    eigVects = eigVects[:,eigVals_index ]
    eigVects_select = eigVects[:, 0:feature_num]
    print(eigVects_select)
    lowDataMat = np.dot(eigVects_select.T, meanRemoved) 
    reconMat = np.dot(eigVects_select, lowDataMat)+meanVals.reshape((2, 1))
    return lowDataMat,reconMat

X_data = np.array([x,y])
lowDataMat, reconMat = pca(X_data, 1)  
plt.scatter(x,y,c = "b")
plt.scatter(reconMat[0,:],reconMat[1,:],c="r")
plt.show()






