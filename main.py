# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/22 15:42  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from LOF import LOF
from LOF_sklearn import LOF_sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data, _ = datasets.make_moons(500, noise=0.1, random_state=1)
k = 20

plt.figure("ours")
lof_list = LOF(data, k)
colors = np.array(['b', 'black'])
result_list = np.zeros(len(data), dtype=int)
result_list[lof_list > 1.2] = 1
plt.scatter(data[:, 0], data[:, 1], c = colors[result_list%2])
outliers_lof = lof_list[result_list == 1]
radius = (outliers_lof.max() - outliers_lof) / (outliers_lof.max() - outliers_lof.min())


plt.scatter(data[result_list==1, 0], data[result_list==1, 1], s=radius * 1000, edgecolors='r',
            facecolors='none', label='Outlier scores')
legend = plt.legend(loc='upper left')

plt.figure("sklearn")
lof_list = LOF_sklearn(data, k)
colors = np.array(['b', 'black'])
result_list = np.zeros(len(data), dtype=int)
result_list[lof_list > 1.2] = 1
plt.scatter(data[:, 0], data[:, 1], c = colors[result_list%2])
outliers_lof = lof_list[result_list == 1]
radius = (outliers_lof.max() - outliers_lof) / (outliers_lof.max() - outliers_lof.min())


plt.scatter(data[result_list==1, 0], data[result_list==1, 1], s=radius * 1000, edgecolors='r',
            facecolors='none', label='Outlier scores')
legend = plt.legend(loc='upper left')

plt.show()