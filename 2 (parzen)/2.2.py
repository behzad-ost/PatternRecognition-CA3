import numpy as np
import matplotlib.pyplot as plt

samples = [-7, -5, -4, -3, -2, 0, 2, 3, 4, 5, 7]

def window_func(x):
    if np.abs(x) < (1/2.):
        return 1
    return 0
    
def parzen(x,sample,h):
    k = 0
    for sample in sample:
        k += window_func((sample-x)/h)
    p=k/float(len(samples))/h
    return p


for H in [0.25,0.5,1,2,3,5]:
    for j in [1,4,11]:
        h=H/np.sqrt(j)
        X=np.linspace(-10,10,1000)
        p=[]
        for x in X:
            p.append(parzen(x,samples,h))
        plt.plot(X,p,label='j = '+str(j))
    plt.legend()
    plt.savefig('p2.2.h('+str(H)+').png',dpi=600)
    plt.show()