import numpy as np
import matplotlib.pyplot as plt

samples = [-7, -5, -4, -3, -2, 0, 2, 3, 4, 5, 7]

def window_func(x):
    if np.abs(x) < (1/2.):
        return 1
    return 0

def parzen(x,samples,h):
    k = 0 
    for sample in samples:

        k+=window_func((sample-x)/h)
    p=k/float(len(samples))/h
    return p


for j in [1,4,11]:
    h=1/float(np.sqrt(j))
    X=np.linspace(-10,10,10000)
    p=[]
    for x in X:
        p.append(parzen(x,samples,h))
    plt.plot(X,p,label='j = '+str(j))

plt.legend()
plt.savefig('2.1.png',dpi=600)
plt.show()