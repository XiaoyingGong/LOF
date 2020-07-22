# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/22 11:11  
# IDE：PyCharm 
# des: The implementation of LOF
# des: a naive implementation, do not concern the computational efficient
# input(s)：
# output(s)：

from sklearn.neighbors import KDTree
import numpy as np

# des: training a kd_tree
def train_kd_tree(train_data):
    kd_tree = KDTree(train_data)
    return kd_tree

# input:
# kd_tree: the trained kd_tree for search;
# center: the center point; k: the number of neighbors
# output:
# neighbors_idx: the index of k neighbors.
def select_k_neighbors(center, k, kd_tree):
    center = center.reshape(-1, len(center))
    neighbors_dist, neighbors_idx = kd_tree.query(center, k=k+1)
    return neighbors_idx[0, 1:], neighbors_dist[0, 1:]

def k_distance(p, k, kd_tree):
    idx, dist = select_k_neighbors(p, k, kd_tree)
    k_distance = dist[-1]
    return k_distance

def rechability_distance(p, o, k, kd_tree):
    k_dist = k_distance(o, k, kd_tree)
    euclidean_dist = np.linalg.norm(p - o)
    reach_dist = np.maximum(k_dist, euclidean_dist)
    return reach_dist

def local_rechability_density(p, k, kd_tree, data):
    # searching neighborhoods
    idx, dist = select_k_neighbors(p, k, kd_tree)
    neighbor_points = data[idx]
    sum_reach_dist = 0
    for i in range(k):
        sum_reach_dist += rechability_distance(p, neighbor_points[i], k, kd_tree)
    lrd = (sum_reach_dist / k) ** (-1)
    return lrd

def local_outlier_factor(p, k, kd_tree, data):
    idx, dist = select_k_neighbors(p, k, kd_tree)
    neighbor_points = data[idx]
    lrd_p = local_rechability_density(p, k, kd_tree, data)
    lrd_o_sum = 0
    for i in range(k):
        lrd_o_sum += local_rechability_density(neighbor_points[i], k, kd_tree, data)
    lof = lrd_o_sum / (k * lrd_p)
    return lof

def LOF(data, k):
    data_len = len(data)
    lof_list = np.zeros(data_len)
    # constructing kd-tree
    kd_tree = train_kd_tree(data)
    for i in range(data_len):
        lof_list[i] = local_outlier_factor(data[i], k, kd_tree, data)
    return lof_list

