import numpy as np

x = np.array([[1, 1],[1, 2],[2, 3],[3, 4],[3, 5]])
y = [0,1,1,0,0]
w = np.zeros(shape = (2,))

def sigmoid(m):
    return 1/(1+np.e**-m)
def H_(w, x):
    return np.dot(x, w)
def y_predict(w, x):
    return sigmoid(H_(w, x))
def loss(y_predict, I):
    return np.sum((y*np.log(y_predict(w, x))) + ((I-y) * (I-np.log(y_predict(w, x)))))
def err(y,y_predict):
    return y-y_predict(w. x)
def gradient(err, x):
    return np.dot(x.T, err)

l=loss(y_predict,y)
while(l>1e-3):
    w -=0.01*gradient(err, x)
    l=loss(y_predict, y)
    err1 = err(y, y_predict)
    print(w)

