# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:52:51 2020

@author: Mete
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%create dataset
#class1
#gaussian veriler oluşturur(25-5=20, 25+5=30) arasında 100 tane veri oluşturur
x1=np.random.normal(25,5,100)
y1=np.random.normal(25,5,100)

#class2
x2=np.random.normal(55,5,100)
y2=np.random.normal(60,5,100)

#class3
x3=np.random.normal(55,5,100)
y3=np.random.normal(15,5,100)

#sınıflarımızı belirledikten sonra eksenleri birleştirme işlemleri yapıyoruz
x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={"x":x,"y":y}

data=pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()



#%%dendogram

from scipy.cluster.hierarchy import linkage,dendrogram

#cluster içindeki yayılımları minimalize etmek için ward kullanmalıyız
merg=linkage(data,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclieden distance")
plt.show()



#%%hierarchial with sklearn
from sklearn.cluster import AgglomerativeClustering

"""
Bir önceki adımda grafiğe bakarak n_cluster sayısını neden 3 seçtiğimizi anlıyoruz
"""
hiyerartical_cluster=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
cluster=hiyerartical_cluster.fit_predict(data)

data["label"]=cluster

plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")
plt.show()





