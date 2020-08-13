# author: 龚潇颖(Xiaoying Gong)
# date： 2020/8/13 9:15  
# IDE：PyCharm 
# des: 此代码直接将|N_k| == k 其实|N_k|不一定 == k 可能在边缘上有别的点
# input(s)：
# output(s)：
import numpy as np
import copy
import time
def cal_point_2_point_Euclidean(X, Y):
    X_re = X.reshape(-1, 1, len(X[0]))
    X_new = np.tile(X_re, (1, len(X), 1))
    Y_re = Y.reshape(1, -1, len(Y[0]))
    Y_new = np.tile(Y_re, (len(Y), 1, 1))
    return np.sqrt(np.sum((X_new - Y_new)**2, axis=2))

def cal_distance(data):
    X = data
    Y = data
    dist = cal_point_2_point_Euclidean(X, Y)
    return dist

def cal_k_distance(data, k):
    N, _ = data.shape
    dist = cal_distance(data)

    sorted_idx = np.argsort(dist, axis=1)
    dist_sorted = np.sort(dist, axis=1)

    k_neighbors_idx = sorted_idx[:, 1:k+1]
    k_dist = dist_sorted[:, k]
    return k_dist, k_neighbors_idx, dist, dist_sorted, sorted_idx

def cal_reachability_distance(k_neighbors_idx, k_dist, dist):
    N = len(k_neighbors_idx)
    # 第i个点与其邻域的距离
    i_neighbors_dist = np.array([dist[i][k_neighbors_idx[i]] for i in range(N)])
    # 第i个点的邻域的k-distance
    i_k_dist = np.array([k_dist[k_neighbors_idx[i]] for i in range(N)])
    # 计算出可达距离
    reachability_distance = np.maximum(i_neighbors_dist, i_k_dist)
    return reachability_distance

def cal_local_reachability_density(data, k):
    k_dist, k_neighbors_idx, dist, dist_sorted, sorted_idx = \
        cal_k_distance(data, k)
    reachability_distance = cal_reachability_distance(k_neighbors_idx, k_dist, dist)
    local_reachability_density = k / np.sum(reachability_distance, axis=1)
    return local_reachability_density, k_neighbors_idx

def cal_local_outlier_factor(data, k):
    N, d = data.shape
    local_reachability_density, k_neighbors_idx = \
        cal_local_reachability_density(data, k)
    # 计算i_p的邻域的local_reachability_density的和
    i_neighbors_lrd = np.array([local_reachability_density[k_neighbors_idx[i]] for i in range(N)])
    LOF = np.sum(i_neighbors_lrd, axis=1) / (k * local_reachability_density)
    return LOF

if __name__ == '__main__':
    from sklearn import preprocessing
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    np.random.seed(10)
    data = np.random.multivariate_normal([1, 1], [[2, 0], [0, 2]], 2000)

    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    k = 10
    time_1 = time.time()
    lof = cal_local_outlier_factor(data, k)
    print("lof:", lof)
    time_2 = time.time()
    print(time_2 - time_1)
    idx = np.argwhere(lof > 1.5).flatten()
    print(idx)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data[idx, 0], data[idx, 1], c='red')
    plt.show()
