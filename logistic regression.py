import numpy as np

x = np.array([[1, 1],[1, 2],[2, 3],[3, 4],[3, 5]])
y = np.array([0,1,1,0,0])
w = np.zeros(shape = (2,))

def sigmoid(m):
    return 1/(1+np.e**-m)
def H_(w, x):
    return np.dot(x, w)
def y_predict(w, x):
    return sigmoid(H_(w, x))
def loss(y_predict):
    return - np.sum((y*np.log(y_predict(w, x))) + ((1-y) * (np.log(1-y_predict(w, x)))))
def err(y,y_predict):
    return y-y_predict(w, x)
def gradient(err, x):
    return np.dot(x.T, err)

l=loss(y_predict)
err1 = err(y, y_predict)
while(l>1e-3):
    w -=0.01*gradient(err1, x)
    l=loss(y_predict)
    err1 = err(y, y_predict)
    print(w)
print(w)
