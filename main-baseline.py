import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon as shapePolygon
from matplotlib.patches import Polygon as matPolygon
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import itertools
import networkx as nx
from queue import PriorityQueue
import math
import copy
import time
from matplotlib.font_manager import FontProperties
from itertools import combinations
from scipy.spatial import Delaunay
from memory_profiler import profile

start_time = time.time()

gmin_pre = 0.3
R = 30.0
font = FontProperties(fname="C:/Windows/Fonts/STXINGKA.TTF")

# 创建颜色映射对象
allInstance_positionDict = {}  # 所有实例对应的位置-字典
allPosition_instanceDict = {}  # 所有位置对应的实例-字典
allInstancePosition = []  # 所有实例的位置
feature_dict = {}  # 记录特征以及其实例：特征-实例
f = open("data/datas1.CSV")  # 返回一个文件对象
line = f.readline()  # 调用文件的 readline()方法
x = {}  # 绘制点图的x坐标
y = {}  # 绘制点图的y坐标
allGetMaxClics = {}  # 获取极大团的输入数组
allGetMaxClics_new = []  # 获取极大团的输入数组
region_Polygon = {}  # 频繁区域边界点-区域字典
haveUseRegion = set()
# 真实数据集
# while line:
#     position = []
#     patternRowName = ''
#     if line[0] == 't':
#         line = f.readline()
#     line = line.replace('\n', "")
#     lists = line.split(',')
#     patternRowName = lists[0] + '|' + lists[1]
#     if lists[0] not in feature_dict:
#         feature_dict[lists[0]] = []
#     feature_dict[lists[0]].append(lists[1])
#     if lists[0] not in x:
#         x[lists[0]] = []
#         y[lists[0]] = []
#     if lists[2] != "" and lists[3] != "":
#         position = [float(lists[2]), float(lists[3])]
#         allInstancePosition.append(position)
#         pointPositionName = [position, lists[0]]
#         allGetMaxClics_new.append(pointPositionName)
#         allGetMaxClics[patternRowName] = position
#         position_key = tuple(position)
#         x[lists[0]].append(round(float(lists[2]), 6))
#         y[lists[0]].append(round(float(lists[3]), 6))
#         allInstance_positionDict[patternRowName] = position
#         allPosition_instanceDict[position_key] = patternRowName
#     line = f.readline()
# f.close()

# 合成数据集
flagid = 0
while line:
    if flagid >15000:
        break
    position = []
    patternRowName = ''
    if line[0] == 't':
        line = f.readline()
    line = line.replace('\n', "")
    lists = line.split('\t')
    patternRowName = lists[1] + '|' + lists[0]
    if lists[1] not in feature_dict:
        feature_dict[lists[1]] = []
    feature_dict[lists[1]].append(lists[0])
    if lists[1] not in x:
        x[lists[1]] = []
        y[lists[1]] = []
    if lists[2] != "" and lists[3] != "":
        position = [float(lists[2]), float(lists[3])]
        allInstancePosition.append(position)
        pointPositionName = [position, lists[1]]
        allGetMaxClics_new.append(pointPositionName)
        allGetMaxClics[patternRowName] = position
        position_key = tuple(position)
        x[lists[1]].append(round(float(lists[2]), 6))
        y[lists[1]].append(round(float(lists[3]), 6))
        allInstance_positionDict[patternRowName] = position
        allPosition_instanceDict[position_key] = patternRowName
    line = f.readline()
    flagid += 1
f.close()


# 读取频繁模式的行实例
patternName_readRow = ''  # 在读取行实例时记录模式的名
patterns_region_rowDict = {}  # 模式-区域-行实例字典
pattern_rowInstancesDict = {}  # 模式-模式行实例（模式名|id）
clique = set()  # 保存极大团
maxPatternLen = 0

# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists

def PartitionBasedBK(instances, r):
    global clique
    global maxPatternLen
    S = sorted(list(instances.keys()))
    while S:
        N = set()
        for key in S:
            if key != S[0] and is_neighbor(instances[key], instances[S[0]], r):
                N.add(key)

        N.add(S[0])
        neighbor = set(N)
        for name in N:
            for key in S:
                if name != key and is_neighbor(instances[name], instances[key], r):
                    neighbor.add(key)
        N_neighbor = list(sorted(neighbor))
        neighbor_map = get_neighbor_map(N_neighbor, r)
        KB_piovt_base([], N_neighbor, [], neighbor_map)
        for s in N:
            S.remove(s)
    return clique


def is_neighbor(a, b, R):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return math.sqrt(x * x + y * y) <= R


def get_neighbor_map(N, r):
    neighbor_map = {}
    for n1 in N:
        neighbors = []
        for n2 in N:
            if n1 != n2 and is_neighbor(allGetMaxClics[n1], allGetMaxClics[n2], r):
                neighbors.append(n2)
        neighbor_map[n1] = neighbors
    return neighbor_map

def KB_piovt_base(Q, P, X, neighbor_map):
    union = set(P) | set(X)
    if not union:
        if len(Q) > 1:
            allSame = True
            for instance in Q:
                if instance[0] != Q[0][0]:
                    allSame = False
                    break
            if not allSame:
                Q.sort()
                clique.add(tuple(Q))
        return

    if Q:
        ccmc_setstr = True
        allSameFeatureQP = set()
        for instance in P:
            allSameFeatureQP.add(instance.split('|')[0])
        if len(allSameFeatureQP)!=1:
            ccmc_setstr = False
        else:
            for instance in Q:
                allSameFeatureQP.add(instance.split('|')[0])
            if len(allSameFeatureQP)==1:
                ccmc_setstr = False
        if ccmc_setstr:
            union_QP = list(set(P) | set(Q))
            union_QP.sort()
            clique.add(tuple(union_QP))
            return

    max = -1
    pivot = ""
    for instance in union:
        if len(neighbor_map[instance]) > max:
            max = len(neighbor_map)
            pivot = instance
    pivot_neighbors = neighbor_map[pivot].copy()
    new_candidate_p = list(set(P) - set(pivot_neighbors))

    pq = PriorityQueue()
    for v in new_candidate_p:
        pq.put((len(neighbor_map[v]), v))

    while not pq.empty():
        v = pq.get()[1]
        v_neighbors = copy.deepcopy(neighbor_map[v])
        new_Q = Q + [v]
        KB_piovt_base(new_Q, list(set(P) & set(v_neighbors)), list(set(X) & set(v_neighbors)), neighbor_map)
        P.remove(v)
        X.append(v)

# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(dists):
    '''算法1'''
    # N = np.shape(dists)[0]
    # tt = np.reshape(dists, N * N)
    # percent = 2.0
    # position = int(N * (N - 1) * percent / 100)
    # dc = np.sort(tt)[position + N]

    ''' 算法 2 '''
    N = np.shape(dists)[0]
    max_dis = np.max(dists)
    min_dis = np.min(dists)
    dc = (max_dis + min_dis) / 2

    while True:
        n_neighs = np.where(dists < dc)[0].shape[0] - N
        rate = n_neighs / (N * (N - 1))

        if rate >= 0.01 and rate <= 0.02:
            break
        if rate < 0.01:
            min_dis = dc
        else:
            max_dis = dc

        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 通过阈值选取 rho与delta都大的点
# 作为聚类中心
def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


# 选取 rho与delta乘积较大的点作为
# 聚类中心
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


# 求极大团
def getMaxClique(points_with_categories):
    # 定义阈值和无穷大的距离值
    infinite_distance = math.inf
    print("进入极大团挖掘函数：" + str(time.time() - start_time))
    # 创建无向图
    G = nx.Graph()

    # 添加点集中的所有点作为图的节点，并设置类别属性
    for i, (point, category) in enumerate(points_with_categories):
        G.add_node(i, pos=tuple(point), category=category)

    # 计算所有点两两之间的距离，并添加边（距离小于阈值时才添加）
    for u, v in itertools.combinations(G.nodes(), 2):
        u_pos = G.nodes[u]['pos']
        v_pos = G.nodes[v]['pos']
        distance = ((u_pos[0] - v_pos[0]) ** 2 + (u_pos[1] - v_pos[1]) ** 2) ** 0.5

        u_category = G.nodes[u]['category']
        v_category = G.nodes[v]['category']

        # 如果两个点属于同一类别，则将它们之间的距离设置为无穷大
        if u_category == v_category:
            distance = infinite_distance

        if distance <= R:
            G.add_edge(u, v, weight=distance)
    print("查找前：" + str(time.time() - start_time))
    # 使用Bron-Kerbosch算法找到所有极大团
    maximal_cliques = list(nx.find_cliques(G))
    print("查找后：" + str(time.time() - start_time))
    # 筛选极大团，排除只包含一个节点的团
    filtered_cliques = [clique for clique in maximal_cliques if len(clique) > 1]
    # 假设 filtered_cliques 是之前求得的极大团列表
    # resultList 用于存储用名称替换的结果
    print(len(filtered_cliques))
    resultList = []
    for clique in filtered_cliques:
        new_clique = [allPosition_instanceDict.get(tuple(points_with_categories[i][0]), None) for i in clique]
        # 使用列表推导式将坐标对替换为名称，并添加到 resultList 中
        resultList.append(new_clique)
    # 将结果转换为坐标形式输出
    # resultList = [[points_with_categories[i][0] for i in clique] for clique in filtered_cliques]
    return resultList


allMaxClics_Hash = {}  # 所有极大团的哈希表结构 模式名-特征-特征id
allMaxClics_Dict = {}  # 所有极大团的快速查找 模式-极大团
# allMaxCliques = getMaxClique(allGetMaxClics)  # 所有极大团
allMaxCliques = PartitionBasedBK(allGetMaxClics, R)
print("极大团获取时间： " + str(time.time() - start_time))
# print(allMaxCliques)
candidatePattern_father = {}  # 记录候选模式的父级，方便后面计算PI
candidatePattrn_global = {}  # 全局候选模式
candidatePattrn_local = {}  # 区域候选模式
prevalentPattern_region = {}  # 区域频繁模式及其区域
prevalentPattern_global = []

maxPatternLen = 0
for maxClics in allMaxCliques:
    patternName = set()
    # print(maxClics)
    for instance_1 in maxClics:
        patternName.add(instance_1.split('|')[0])
    if len(patternName) > maxPatternLen:
        maxPatternLen = len(patternName)
    patternName = ' '.join(sorted(list(patternName)))
    if patternName not in allMaxClics_Dict:
        allMaxClics_Dict[patternName] = []
    # print(patternName)
    allMaxClics_Dict[patternName].append(sorted(maxClics))
    if patternName not in allMaxClics_Hash:
        allMaxClics_Hash[patternName] = {}
        for instance_2 in maxClics:
            allMaxClics_Hash[patternName][instance_2.split('|')[0]] = set()
    for instance_1 in maxClics:
        # if instance_1.split('|')[1] not in allMaxClics_Hash[patternName][instance_1.split('|')[0]]:
        allMaxClics_Hash[patternName][instance_1.split('|')[0]].add(instance_1.split('|')[1])
print(len(allMaxClics_Dict))

# 清除极大实例团的内存
allMaxCliques = None

# patternInMaxClique = {}  # 记录模式的行实例在那些极大团里面
allPatternParticipation = {}  # 所有模式的参与实例字典
# allParticipation_position = {}  # 所有模式的参与实例的坐标
pattern_rowInstancesDict = {}  # 所有模式的行实例
patternInMaxClique_reduce = {}  # 记录模式的行实例在那些极大团里面--精简版
print("极大团字典建立时间： " + str(time.time() - start_time))
# 初始化全局候选模式字典
# 在此处就可以获得模式的参与实例、行实例、在那些极大实例团当中
for i in range(2, maxPatternLen + 1):
    candidatePattrn_local[i] = []
    candidatePattrn_global[i] = []
for patternName in allMaxClics_Hash:
    patternName_list = patternName.split()
    if len(patternName_list) == 2:
        if patternName not in candidatePattrn_global[2]:
            candidatePattrn_global[2].append(patternName)
    # 生成所有大于2个元素的子集
    subsets = [subset for r in range(2, len(patternName_list) + 1) for subset in combinations(patternName_list, r)]
    for subset in subsets:
        patternName_son = ' '.join(subset)
        # if patternName_son not in patternInMaxClique:
        #     patternInMaxClique[patternName_son] = []
        if patternName_son not in patternInMaxClique_reduce:
            patternInMaxClique_reduce[patternName_son] = set()
        if patternName_son not in allPatternParticipation:
            allPatternParticipation[patternName_son] = {}
        # if patternName_son not in allParticipation_position:
        #     allParticipation_position[patternName_son] = []
        # if patternName_son not in pattern_rowInstancesDict:
        #     pattern_rowInstancesDict[patternName_son] = set()
        # 获取模式的参与实例以及坐标集 ---- 基本不耗时
        for feature in subset:
            if feature not in allPatternParticipation[patternName_son]:
                allPatternParticipation[patternName_son][feature] = set()
            for instance_id in allMaxClics_Hash[patternName][feature]:
                # if instance_id not in allPatternParticipation[patternName_son][feature]:
                # allParticipation_position[patternName_son].append(
                #     allInstance_positionDict[feature + "|" + str(instance_id)])
                allPatternParticipation[patternName_son][feature].add(instance_id)
        # print("获取参与实例：")
        # print(time.time() - start_time)
        patternInMaxClique_reduce[patternName_son].add(patternName + "|" + str(len(allMaxClics_Dict[patternName])))
        # for maxClic in allMaxClics_Dict[patternName]:
        #     # patternInMaxClique[patternName_son].append(tuple(maxClic))
        #     row_list = []
        #     for point in maxClic:
        #         if point.split('|')[0] in subset:
        #             row_list.append(point)
        #     pattern_rowInstancesDict[patternName_son].add(tuple(row_list))
        #     # if row_list not in pattern_rowInstancesDict[patternName_son]:
        #     #     pattern_rowInstancesDict[patternName_son].append(row_list)
        # print("获取行实例：")
        # print(time.time() - start_time)
        if patternName_son not in candidatePattrn_global[len(subset)]:
            candidatePattrn_global[len(subset)].append(patternName_son)

# 回收内存
allMaxClics_Hash = None
allMaxClics_Dict = None
print("参与实例、行实例获取时间： " + str(time.time() - start_time))

print("全局候选模式数目：" + str(len(allPatternParticipation)))
# 网格划分计算频繁系数，并且挖掘区域频繁模式
allregion = 0
for k in candidatePattrn_global:
    for patternName in candidatePattrn_global[k]:
        PI_Global = 1
        for featureName in patternName.split(' '):
            PR_Global = float(len(allPatternParticipation[patternName][featureName])) / float(
                len(feature_dict[featureName]))
            if PR_Global < PI_Global:
                PI_Global = PR_Global
        if PI_Global < gmin_pre:  # 满足集中分布且行实例数大于等于2
            continue
        else:
            prevalentPattern_global.append(patternName)
            del allPatternParticipation[patternName]
            # del allParticipation_position[patternName]
print("区域候选获取时间： " + str(time.time() - start_time))
print("区域候选模式数目：" + str(len(allPatternParticipation)))


# 挖掘代表性模式的频繁区域
def minePatternRegion():
    for rePatternName in allPatternParticipation:
        feature_dict_copy = copy.deepcopy(feature_dict)
        rePatternName_list = rePatternName.split(" ")
        for feature in rePatternName_list:
            for participation in allPatternParticipation[rePatternName][feature]:
                feature_dict_copy[feature].remove(participation)
        # 创建并拟合DBSCAN模型
        dbscan = DBSCAN(eps=1.5 * R, min_samples=len(rePatternName_list) - 1)
        allParticipation_position = set()
        for feature in rePatternName_list:
            for instance_id in allPatternParticipation[rePatternName][feature]:
                allParticipation_position.add(tuple(allInstance_positionDict[feature + "|" + str(instance_id)]))
        allParticipation_positionNP = np.array(list(allParticipation_position))
        dbscan.fit(allParticipation_positionNP)

        # 获取预测的簇标签
        labels = dbscan.labels_

        # 获取簇数
        # num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # print("Number of Clusters:", num_clusters)

        # 将点分组到不同的簇中
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(allParticipation_positionNP[i].tolist())
        test = ["A E F G O"]
        if rePatternName in test:
            print(clusters)
        # 遍历每个簇中(每个候选区域）
        for label, points in clusters.items():
            if len(points) <= 2:  # 只有两个点无法构成区域
                continue
            participatingInInstances = {}  # 该区域的参与实例
            try:
                hull_D = Delaunay(np.array(points))
            except:
                continue
            for point in points:
                if allPosition_instanceDict[tuple(point)].split('|')[0] not in participatingInInstances:
                    participatingInInstances[allPosition_instanceDict[tuple(point)].split('|')[0]] = []
                participatingInInstances[allPosition_instanceDict[tuple(point)].split('|')[0]].append(
                    allPosition_instanceDict[tuple(point)].split('|')[1])
            # print(participatingInInstances)
            PI = 1
            for featureName in rePatternName_list:
                featureInstanceInNum = len(participatingInInstances[featureName])
                deleteId = []
                for instance_id in feature_dict_copy[featureName]:
                    if hull_D.find_simplex(allInstance_positionDict[featureName + '|' + str(instance_id)]) >= 0:
                        deleteId.append(instance_id)
                        featureInstanceInNum += 1
                PR = float(len(participatingInInstances[featureName])) / float(featureInstanceInNum)
                if PR < PI:
                    PI = PR
                for id in deleteId:
                    feature_dict_copy[featureName].remove(id)
            if PI > gmin_pre:
                if rePatternName not in patterns_region_rowDict:
                    patterns_region_rowDict[rePatternName] = {}
                if rePatternName not in prevalentPattern_region:
                    prevalentPattern_region[rePatternName] = []
                hull = ConvexHull(points)
                region_list = hull.points[hull.vertices]
                # 将数据列表转换为元组列表
                region_tuple = list(tuple(item) for item in region_list)
                region_tuple.append(tuple(region_list[0]))
                region_tuple = tuple(region_tuple)
                prevalentPattern_region[rePatternName].append(region_tuple)
                patterns_region_rowDict[rePatternName][region_tuple] = []
                # for rowInstance in pattern_rowInstancesDict[rePatternName]:
                #     # rowInstance_id = []
                #     if hull_D.find_simplex(allInstance_positionDict[rowInstance[0]]) >= 0:
                #         # for instance in rowInstance:
                #         #     rowInstance_id.append(instance.split('|')[1])
                #         patterns_region_rowDict[rePatternName][region_tuple].append(rowInstance)
            # else:
            #     for rowInstance in pattern_rowInstancesDict[rePatternName]:
            #         if hull_D.find_simplex(allInstance_positionDict[rowInstance[0]]) >= 0:
            #             delMaxClics = []
            #             for maxInClic in patternInMaxClique[rePatternName]:
            #                 if tuple(rowInstance) in maxInClic:
            #                     delMaxClics.append(maxInClic)
            #             for delMaxClic in delMaxClics:
            #                 patternInMaxClique[rePatternName].remove(delMaxClic)


# 挖掘所有区域频繁模式
minePatternRegion()
print("区域频繁同位模式数目："+str(len(patterns_region_rowDict)))
for patternName in allPatternParticipation:
    if patternName not in patterns_region_rowDict:
        print(str(patternName), end=", ")


# 判断行实例是否在极大团当中
def is_subset(subset, x):
    for sublist in x:
        if set(sublist).issubset(set(subset)):
            return True
    return False


patternNameList = []
for nameKey in patterns_region_rowDict:
    patternNameList.append(nameKey)
n = len(patterns_region_rowDict)
# print("极大团压缩后数量:"+str(len(patternInMaxClique_reduce)))
haveMaxClics_num = {}  # 记录模式极大实例团的数目
# 假设有一个模式名到模式编号的映射字典 pattern_mapping 如下
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        intersection_set = patternInMaxClique_reduce[patternNameList[i]] & patternInMaxClique_reduce[patternNameList[j]]
        numerator = 0
        # print(patternNam
        for inter in intersection_set:
            numerator += float(inter.split("|")[1])
        pattern_1 = 0
        pattern_2 = 0
        if patternNameList[i] not in haveMaxClics_num:
            for patternIn in patternInMaxClique_reduce[patternNameList[i]]:
                pattern_1 += float(patternIn.split("|")[1])
            haveMaxClics_num[patternNameList[i]] = pattern_1
        else:
            pattern_1 = haveMaxClics_num[patternNameList[i]]

        if patternNameList[j] not in haveMaxClics_num:
            for patternIn in patternInMaxClique_reduce[patternNameList[j]]:
                pattern_2 += float(patternIn.split("|")[1])
            haveMaxClics_num[patternNameList[j]] = pattern_2
        else:
            pattern_2 = haveMaxClics_num[patternNameList[j]]
        lowNum = (math.sqrt(pattern_1) * math.sqrt(pattern_2))
        if lowNum == 0:
            SIM = 0
        else:
            SIM = numerator / lowNum
        DIS_SIM = 1 - SIM
        distance_matrix[i][j] = DIS_SIM
        distance_matrix[j][i] = DIS_SIM

# # 使用层次聚类算法进行聚类
# Z = linkage(distance_matrix, method='average')  # 层次聚类的链接矩阵
# threshold = 0.8  # 设置一个合适的距离阈值
# labels = fcluster(Z, threshold, criterion='distance')  # 根据距离阈值划分簇

# 使用DPC聚类算法进行聚类
# 计算距离矩阵
dists = getDistanceMatrix(distance_matrix)
# 计算dc
dc = select_dc(dists)
# print("dc", dc)
# 计算局部密度
rho = get_density(dists, dc, method="Gaussion")
# 计算密度距离
deltas, nearest_neiber = get_deltas(dists, rho)
# 获取聚类中心点
# centers = find_centers_K(rho, deltas, 3)
centers = find_centers_auto(rho, deltas)
# print("centers", centers)
labels = cluster_PD(rho, centers, nearest_neiber)
cluster_patternDict = {}  # 簇——模式字典
clusterAttributeDict_threshold = {}  # 簇-属性（通过阈值生成的簇）
clusterAttributeDict_all = {}  # 簇-属性（层次树所有的属性范围）
# 输出每个凸包所属的簇
for i, label in enumerate(labels):
    if label not in cluster_patternDict:
        cluster_patternDict[label] = []
    cluster_patternDict[label].append(patternNameList[i])
# 按照键对字典排序并得到有序字典
cluster_patternDict = {k: v for k, v in sorted(cluster_patternDict.items())}
end_time = time.time()
print("模式聚类时间： " + str(time.time() - start_time))
print("模式聚类后的簇数目：" + str(len(cluster_patternDict)))
print(cluster_patternDict)


# print("代表性模式挖掘时间： " + str(time.time() - start_time))
with open('output.csv', 'w') as file:
    for patternName in patterns_region_rowDict:
        # print(patternName)
        # 打开或创建一个CSV文件
        file.write(','.join(str(ord(letter)-65) for letter in patternName.split()) + '\n')
#     for region in patterns_region_rowDict[patternName]:
#         print(region)
# print(cluster_patternDict)
# print("代表模式数目：" + str(len(patterns_region_rowDict)))


def merge_integras(integras):
    merged = []
    while integras:
        # 取出列表中的第一个元素
        first, *rest = integras
        first = set(first)
        changed = True
        while changed:
            changed = False
            rest2 = []
            for integra in rest:
                if first & set(integra):  # 如果有交集
                    first |= set(integra)  # 合并它们
                    changed = True
                else:
                    rest2.append(integra)  # 否则保留该列表
            rest = rest2
        merged.append(list(first))
        integras = rest
    return merged


def regionalIntegration(regions):
    global region_Polygon
    integras = []
    integras_set = set()
    for i in range(0, len(regions)):
        if regions[i] in region_Polygon:
            polygon1 = region_Polygon[regions[i]]
        else:
            polygon1 = Polygon(regions[i])
            region_Polygon[regions[i]] = polygon1
        for j in range(i + 1, len(regions)):
            if regions[j] in region_Polygon:
                polygon2 = region_Polygon[regions[j]]
            else:
                polygon2 = Polygon(regions[j])
                region_Polygon[regions[j]] = polygon2
            # polygon1 = Polygon(regions[i])
            # polygon2 = Polygon(regions[j])
            overlap = polygon1.intersects(polygon2)
            toch = polygon1.touches(polygon2)
            if overlap and toch == False:
                integras_set.add(i)
                integras_set.add(j)
                flag = 0
                for integra in integras:
                    if i in integra:
                        integra.append(j)
                        flag = 1
                        break
                    elif j in integra:
                        integra.append(i)
                        flag = 1
                        break
                if flag == 0:
                    integras.append([i, j])
                else:
                    integras = merge_integras(integras)
    if len(integras) == 0:
        return regions
    newRegions = copy.deepcopy(regions)
    for i in range(0, len(regions)):
        if i in integras_set:
            newRegions.remove(regions[i])
    for integra in integras:
        polygon1 = region_Polygon[regions[integra[0]]]
        for i in range(1, len(integra)):
            polygon2 = region_Polygon[regions[integra[i]]]
            polygon1 = polygon1.union(polygon2)
        newRegions.append(tuple(polygon1.exterior.coords))
        region_Polygon[tuple(polygon1.exterior.coords)] = polygon1
    return newRegions


def getRepresentativeness(points1, points2):
    global haveUseRegion, region_Polygon, time_for, time_area
    intersect = []
    for key1 in points1:
        useFlag = 0
        if key1 in region_Polygon:
            polygon1 = region_Polygon[key1]
        else:
            polygon1 = Polygon(key1)
            region_Polygon[key1] = polygon1
        for key2 in points2:
            if key2 in region_Polygon:
                polygon2 = region_Polygon[key2]
            else:
                polygon2 = Polygon(key2)
                region_Polygon[key2] = polygon2
            overlap = polygon1.intersects(polygon2)
            if overlap:
                useFlag = 1
                inter = polygon1.intersection(polygon2)
                if isinstance(inter, Polygon) == 0:
                    continue
                intersect.append(tuple(inter.exterior.coords))
                region_Polygon[tuple(inter.exterior.coords)] = inter
        if useFlag:
            haveUseRegion.add(key1)
    area1 = 0  # 被覆盖的频繁区域总面积
    area2 = 0  # 重叠区域的总面积
    # 计算被覆盖模式面积
    for region in points2:
        polygonx = region_Polygon[region]
        area1 += polygonx.area
    # 计算重叠区域面积
    for region in intersect:
        polygonx = region_Polygon[region]
        area2 += polygonx.area
    return area2 / area1


allInterTime = 0
allthresholdTime = 0
representativePattern_dict = {}
# 挖掘各个簇中的代表性模式及其频繁区域
for key in cluster_patternDict:
    representativePattern = []  # 保存候选代表性模式
    region_Polygon.clear()
    if len(cluster_patternDict[key]) == 0:
        break
    elif len(cluster_patternDict[key]) == 1:
        representativePattern.append(cluster_patternDict[key][0])
    else:
        candidateRepPattern = list(copy.deepcopy(cluster_patternDict[key]))  # 簇中的模式
        haveRePattern_remove = []  # 代表或者被代表模式
        rePattern_Region = []
        while len(candidateRepPattern) != 0:
            maxLen = 0
            rePattern = ""
            for patternIncluster in candidateRepPattern:
                if len(patternIncluster.split(" ")) >= maxLen:
                    rePattern = patternIncluster
                    maxLen = len(patternIncluster.split(" "))
            representativePattern.append(rePattern)
            candidateRepPattern.remove(rePattern)
            for regionx in patterns_region_rowDict[rePattern]:
                rePattern_Region.append(regionx)
            if len(rePattern_Region) >= 2:
                regionalIntegration_time = time.time()
                rePattern_Region = regionalIntegration(rePattern_Region)
                allInterTime += (time.time()-regionalIntegration_time)
            # print(rePattern)
            haveUseRegion = set()
            for patternIncluster in candidateRepPattern:
                threshold_get = getRepresentativeness(rePattern_Region, patterns_region_rowDict[patternIncluster])
                if threshold_get >= 0.95:
                    haveRePattern_remove.append(patternIncluster)
            rePattern_Region = list(haveUseRegion)
            # print("计算覆盖度时间：" + str(time.time() - threshold_time))
            candidateRepPattern = [x for x in candidateRepPattern if x not in haveRePattern_remove]
        # 去除掉存在两个子集的代表性模式
        haveRePattern_remove = []
        for rePattern in representativePattern:
            patternName_list = rePattern.split()
            subsets = [subset for r in range(len(patternName_list)-1, len(patternName_list)) for subset in combinations(patternName_list, r)]
            flag = 0
            for subset in subsets:
                patternName_son = ' '.join(subset)
                if patternName_son in representativePattern:
                    flag += 1
                if flag >= 2:
                    break
            if flag >= 2:
                haveRePattern_remove.append(rePattern)
        representativePattern = [x for x in representativePattern if x not in haveRePattern_remove]
        representativePattern_dict[key] = representativePattern

print("代表性模式挖掘时间： " + str(time.time() - start_time))
print(representativePattern_dict)
rePattern_num = 0
for key in representativePattern_dict:
    rePattern_num += len(representativePattern_dict[key])
print("代表模式数目：" + str(rePattern_num))
# f = open("data/compare.txt")  # 返回一个文件对象
# line = f.readline()  # 调用文件的 readline()方法
# allPattern = []
# while line:
#     line = line.replace('\n', "")
#     allPattern.append(line)
#     line = f.readline()
# f.close()
# # print(allPattern)
# rePattern_num = 0
# for key in representativePattern_dict:
#     for patternName in representativePattern_dict[key]:
#         # if patternName not in allPattern:
#         print(patternName)