import numpy as np
from get_data import get_trajectory
import matplotlib.pyplot as plt

a = get_trajectory()
for i in range(len(a)):
    plt.plot(a[i][0],a[i][1])
plt.title('all tra')
plt.show()

data = []
for i in range(len(a)):
    for j in range(len(a[i][0])):
        data.append(np.array([a[i][0][j], a[i][1][j], a[i][-1]]))

data = np.array(data)
plt.scatter(data[:,0],data[:,1])
plt.show()

from sklearn.cluster import KMeans
import numpy as np
import math as m

def get_dis_matrix(data):
    """
    获得邻接矩阵
    :param data: 样本集合
    :return: 邻接矩阵
    """
    nPoint = len(data)
    dis_matrix = np.zeros((nPoint, nPoint))
    for i in range(nPoint):
        for j in range(i + 1, nPoint):
            dis_matrix[i][j] = dis_matrix[j][i] = m.sqrt(np.power(data[i] - data[j], 2).sum())
    return dis_matrix


def getW(data, k):
    """
    利用KNN获得相似矩阵
    :param data: 样本集合
    :param k: KNN参数
    :return:
    """
    dis_matrix = get_dis_matrix(data)
    W = np.zeros((len(data), len(data)))
    for idx, each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W[idx][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W


def getD(W):
    """
    获得度矩阵
    :param W:  相似度矩阵
    :return:   度矩阵
    """
    D = np.diag(sum(W))
    return D


def getL(D, W):
    """
    获得拉普拉斯举着
    :param W: 相似度矩阵
    :param D: 度矩阵
    :return: 拉普拉斯矩阵
    """
    return D - W


def getEigen(L):
    """
    从拉普拉斯矩阵获得特征矩阵
    :param L: 拉普拉斯矩阵
    :return:
    """
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:cluster_num]
    return eigvec[:, ix]


def plotRes(data, clusterResult, clusterNum):
    """
    结果可视化
    :param data:  样本集
    :param clusterResult: 聚类结果
    :param clusterNum:  聚类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterResult[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')




cluster_num = 3
KNN_k = 5
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data_ss = ss.fit_transform(data[:, :2])

W = getW(data_ss, KNN_k)
D = getD(W)
L = getL(D, W)
eigvec = getEigen(L)
clf = KMeans(n_clusters=cluster_num)
s = clf.fit(eigvec)
C = s.labels_
plotRes(data, np.asarray(C), 5)
plt.xlabel('Time (s)')
plt.ylabel('Distance (ft)')
plt.savefig('spectal-c=3-k=5.png', format='png')
plt.show()