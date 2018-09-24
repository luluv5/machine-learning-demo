##import matplotlib.pyplot as plt
import numpy as np
x=np.array([[1,1],[1,2],[2,3],[3,4],[3,5]])
y=np.array([1,2,3,4,5])
w=np.zeros(shape=(2,))
x1=np.c_[np.ones(x.shape[0]),x]
w1=np.zeros(shape=(3,))
def print_f(x,w):
    return np.dot(x,w)
def err(x,w,y):
    return np.dot(x,w)-y
def loss(err):
    return np.mean(1/2*(err**2))
loss1=float('inf')
err1=err(x1,w1,y)
while (loss1>1e-3):
    grad=np.dot(x1.T,err1)
    w1-=0.01*grad
    err1=err(x1,w1,y)
    loss1=loss(err1)   
print(w1)
m=print_f(x1,w1)
print(m)

