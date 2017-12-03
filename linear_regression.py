# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#x的个数决定了样本量
x = np.arange(-1,1,0.02)
#y为理想函数
y = 2*np.sin(2*x)+0.5*x**2
#y1为离散的拟合数据
y1 = y + 0.5*(np.random.rand(len(x))-0.5)
one = np.ones((len(x),1))
x2 = x.reshape(x.shape[0],1)
A = np.hstack((x2,one))
C = y1.reshape((y1.shape[0],1))

m=[]
for i in range(7):
    a = x**i
    m.append(a)
A7 = np.array(m).T

def optimal(A,b):
    # B = A.T.dot(b)
    p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
    print(p)
    return p


y2 = A.dot(optimal(A,C))
y7 = A7.dot(optimal(A7,C))
plt.plot(x,y,color='g',linestyle='-',marker='',label='real line')
plt.plot(x,y1,color='k',linestyle='',marker='*',label='fitting dataset')
plt.plot(x,y2,color='b',linestyle='-',marker='',label='fitting line')
plt.plot(x,y7,color='r',linestyle='-',marker='',label='fitting line2')
plt.legend()
plt.show()
