#!/usr/bin/env python
# coding: utf-8

# In[23]:


from matplotlib import pyplot as plt  
import numpy as np  
import math
import os
from mpl_toolkits.mplot3d import Axes3D  


# In[24]:


def Rastrigin():
    x = np.arange(-5, 5, 0.1) #创建等差数组 与np.linspace类似但不相同（arange使用的是步长不是步数） 
    y = np.arange(-5, 5, 0.1)  
    X, Y = np.meshgrid(x, y)
    d = 2
    Z = 10 * d + X ** 2 - 10 * np.cos(2 * np.pi * X) + Y ** 2 - 10 * np.cos(2 * np.pi * Y)
    return X, Y, Z, "Rastrigin function"


# In[25]:


def draw_pic(X,Y,Z,title):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.hot)
    ax.set_title(title)
    plt.show()


# In[26]:


X,Y,Z,title = Rastrigin()
draw_pic(X,Y,Z,title)


# In[27]:





# In[ ]:




