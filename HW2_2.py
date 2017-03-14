import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X = np.random.uniform(0,1,size=(100,2))
a = 0.5
b = 0.6
r = 0.4
Y = (((X[:,0] - a)**2 + (X[:,1] - b)**2) < r**2)

colors = ['red','blue']
plt.scatter(X[:,0],X[:,1],c=Y,cmap=matplotlib.colors.ListedColormap(colors))
plt.show()