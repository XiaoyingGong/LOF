# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/22 16:04  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
# 正常显示中文
from pylab import mpl

def LOF_sklearn(data, k):
    lof = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
    # 使用fit预测值来计算训练样本的预测标签
    # （当LOF用于异常检测时，估计量没有预测，
    # 决策函数和计分样本方法）。
    predict = lof.fit_predict(data)
    lof_list = -1 * lof.negative_outlier_factor_
    return lof_list
