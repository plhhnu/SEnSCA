import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import copy

def getData(fileName='./data1.csv'):
    # 加载数据集并分割，分割出来带标签的数据和不带标签的数据
    data = pd.read_csv(fileName, encoding='utf-8-sig')
    labelData = data[data['label'] == 1] # 标签1数据
    noLabelData = data[data['label'] == 0] # 标签0数据
    X = labelData.loc[:, labelData.columns != 'label']
    Y = labelData['label']
    Xtest = noLabelData.loc[:, labelData.columns != 'label']
    return X, Y, Xtest

def k_Means(X_train):
    # 通过K-means聚类将当前正样本看成一类，计算其中心点
    k_means = KMeans(n_clusters=1)
    k_means.fit(X_train) # 使用训练集训练模型
    center = k_means.cluster_centers_ # 中心点
    return center

def calMinMax(X_train,center,pre):
    """
    根据准确率计算边界值
    :param X_train: 正本的训练数据
    :param Y: 正本的标签
    :param center: 正样本的中心点
    :param pre: 准确率
    :return: 边界值
    """
    distance = euclidean_distances(np.array(X_train), np.array(center)) # 计算正样本的数据集离中心点的距离
    minDisA, maxDisA = min(distance), max(distance) # 范围设定在该区间则为100%
    disLen = maxDisA - minDisA # 已知的区间范围
    minDis = minDisA + (1.0 - pre) * 0.5 * disLen # 最小距离
    maxDis = maxDisA - (1.0 - pre) * 0.5 * disLen # 最大距离
    # 对应的原始的数据也需要修改
    distanceTest1 = copy.deepcopy(distance)
    distanceTest1 = np.where((distanceTest1 >= minDis) & (distanceTest1 <= maxDis), 1, distanceTest1)
    distanceTest1 = np.where(distanceTest1 != 1, 0, distanceTest1)
    distanceTest1 = np.array(distanceTest1).reshape(1,-1)[0]
    return distanceTest1.astype(int)

rootFile = './data/'
filename = "data.csv"
X_train, YLabel, Xtest = getData(fileName=rootFile+filename)
center = k_Means(X_train)
pre = 0.6  # 调整边界值 范围是(0,1.0]
Y = calMinMax(X_train,center,pre)
YTest = calMinMax(Xtest,center,pre)
print("其他没有标签的样本预测为正的个数为 ：",np.sum(np.array(YTest) == 1))
# 保存数据
newData1 = copy.deepcopy(X_train)
newData1['label'] = np.array(Y)
newData2 = copy.deepcopy(Xtest)
newData2['label'] = np.array(YTest)
file = rootFile + "kkkk/"
if not os.path.exists(file):
    os.mkdir(file)
pd.concat([newData1, newData2], axis=0).to_csv(file + filename, index=None)