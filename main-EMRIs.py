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
import itertools
from queue import PriorityQueue
import networkx as nx
import math
import copy
import time
from matplotlib.font_manager import FontProperties
from itertools import combinations
from scipy.spatial import Delaunay
from memory_profiler import profile
from pympler import asizeof

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
allGetMaxClics_new = []
# 读取频繁模式的行实例

patternName_readRow = ''  # 在读取行实例时记录模式的名
patterns_region_rowDict = {}  # 模式-区域-行实例字典
pattern_rowInstancesDict = {}  # 模式-模式行实例（模式名|id）
clique = set()  # 保存极大团
maxPatternLen = 0

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
        if len(allSameFeatureQP) != 1:
            ccmc_setstr = False
        else:
            for instance in Q:
                allSameFeatureQP.add(instance.split('|')[0])
            if len(allSameFeatureQP) == 1:
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


allMaxClics_Hash = {}  # 所有极大团的哈希表结构 模式名-特征-特征id
allMaxClics_Dict = {}  # 所有极大团的快速查找 模式-极大团
# allMaxCliques = getMaxClique(allGetMaxClics_new)  # 所有极大团
allMaxCliques = PartitionBasedBK(allGetMaxClics, R)
size_s = asizeof.asizeof(allMaxCliques)
size_s_mb = size_s / (1024 * 1024)

print(f"Total size of set: {size_s_mb:.2f} MB")
print(len(allMaxCliques))
print("扩展极大行实例获取时间： " + str(time.time() - start_time))
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
size_w = asizeof.asizeof(allMaxClics_Dict)
size_s_mb = size_w / (1024 * 1024)

print(f"Total size of set: {size_s_mb:.2f} MB")

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


# 计算极大团的质点
def calculate_centroid(points):
    n = len(points)
    if n == 0:
        return None

    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)

    centroid_x = sum_x / n
    centroid_y = sum_y / n

    return [centroid_x, centroid_y]


def find_grid_cell(grid, grid_size, origin_x, origin_y, x, y):
    dx = x - origin_x
    dy = y - origin_y
    row = int(dy / grid_size)
    col = int(dx / grid_size)
    return row, col


def dfs_region(matrix, vertex, visited, connected_points):
    visited[vertex] = True
    connected_points.append(vertex)
    for i in range(len(matrix)):
        if matrix[vertex][i] > 0 and not visited[i]:
            dfs_region(matrix, i, visited, connected_points)


def find_connected_graph(matrix):
    if not matrix.any() or not matrix[0].any():
        return []

    n = len(matrix)
    visited = [False] * n
    connected_graphs = []

    for i in range(n):
        if not visited[i]:
            connected_points = []
            dfs_region(matrix, i, visited, connected_points)
            connected_graphs.append(sorted(connected_points))

    return connected_graphs


def euclidean_distance(point1, point2):
    assert len(point1) == len(point2), "两个点的维度必须相同"
    squared_sum = sum((p2 - p1) ** 2 for p1, p2 in zip(point1, point2))
    distance = math.sqrt(squared_sum)
    return distance


# 判断点是否在凸包内
def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def is_valid(grid, visited, row, col):
    rows, cols = len(grid), len(grid[0])
    return 0 <= row < rows and 0 <= col < cols and grid[row][col] and not visited[row][col]


def dfs(grid, visited, row, col, component):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    visited[row][col] = True
    component.append((row, col))

    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if is_valid(grid, visited, nr, nc):
            dfs(grid, visited, nr, nc, component)


def merge_connected_components(grid):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    connected_components = []

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] and not visited[i][j]:
                component = []
                dfs(grid, visited, i, j, component)
                connected_components.append(component)

    return connected_components


# # 正方形网格的边长
# grid_size = R
# allInstancePosition_np = np.array(allInstancePosition)
# # 计算网格的原点（即最小的x和y坐标）
# origin_x = min(allInstancePosition_np[:, 0])
# origin_y = min(allInstancePosition_np[:, 1])
#
# # 计算区域的边界（即最大的x和y坐标）
# max_x = max(allInstancePosition_np[:, 0])
# max_y = max(allInstancePosition_np[:, 1])
#
# # 计算网格的行数和列数
# rows = int(np.ceil((max_y - origin_y) / grid_size)) + 1
# cols = int(np.ceil((max_x - origin_x) / grid_size)) + 1
#
# # 初始化网格
# grid = np.zeros((rows, cols), dtype=int)
# print(rows, cols)
# addx = 0
# addy = 0
print("全局候选模式数目：" + str(len(allPatternParticipation)))
# 网格划分计算频繁系数，并且挖掘区域频繁模式
allregion = 0
for k in candidatePattrn_global:
    for patternName in candidatePattrn_global[k]:

        # grid_rowInstance = np.zeros((rows, cols), dtype=int)  # 记录行实例在那些网格出现了
        # gridPInstanceIn_dict = {}

        # PINum = np.mean(grid_rowInstance)
        # Spinum = np.var(grid_rowInstance, ddof=1)
        # PRSc = Spinum / pow(PINum, 2)
        # eve = 1 - PRSc / (rows * cols)

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


# 判断行实例是否在极大团当中
def is_subset(subset, x):
    for sublist in x:
        if set(sublist).issubset(set(subset)):
            return True
    return False


patternNameList = []
for nameKey in allPatternParticipation:
    patternNameList.append(nameKey)
n = len(allPatternParticipation)
haveMaxClics_num = {}  # 记录模式极大实例团的数目
# 假设有一个模式名到模式编号的映射字典 pattern_mapping 如下
distance_matrix = np.zeros((n, n), dtype=np.float16)
testName = ['A D','A E','A F','D E','D F','E F','A D E','A D F','A E F','D E F','A D E F']
for i in range(n):
    # if patternNameList[i] in testName:
    #     print(patternNameList[i])
    for j in range(i + 1, n):
        intersection_set = patternInMaxClique_reduce[patternNameList[i]] & patternInMaxClique_reduce[patternNameList[j]]
        numerator = 0
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
        # if patternNameList[i] in testName and patternNameList[j] in testName:
        #     print(patternNameList[i]+"----"+patternNameList[j]+": "+str(DIS_SIM))
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
regionXY = []


# 挖掘代表性模式的频繁区域
def minePatternRegion(representativePattern):
    for rePatternName in representativePattern:
        if rePatternName in patterns_region_rowDict:
            continue
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


# 挖掘各个簇中的代表性模式及其频繁区域
representativePattern_dict = {}
for key in cluster_patternDict:
    haveGetAllrepattern_flag = 0  # 标记该簇的代表性模式是否都是区域频繁模式
    while haveGetAllrepattern_flag == 0:
        representativePattern = []  # 保存候选代表性模式
        if len(cluster_patternDict[key]) == 0:
            break
        elif len(cluster_patternDict[key]) == 1:
            representativePattern.append(cluster_patternDict[key][0])
        else:
            candidateRepPattern = list(copy.deepcopy(cluster_patternDict[key]))  # 簇中的模式
            haveRePattern_remove = []  # 代表或者被代表模式
            patternInMaxClique_repattern = set()
            while len(candidateRepPattern) != 0:
                # maxLen = len(candidateRepPattern[0].split(" "))
                maxLen = 0
                rePattern = ""
                for patternIncluster in candidateRepPattern:
                    if len(patternIncluster.split(" ")) >= maxLen:
                        rePattern = patternIncluster
                        maxLen = len(patternIncluster.split(" "))
                representativePattern.append(rePattern)
                candidateRepPattern.remove(rePattern)
                patternInMaxClique_repattern.update(patternInMaxClique_reduce[rePattern])
                for patternIncluster in candidateRepPattern:
                    andMaxClique = patternInMaxClique_repattern & patternInMaxClique_reduce[patternIncluster]
                    if len(andMaxClique) == len(patternInMaxClique_reduce[patternIncluster]):
                        haveRePattern_remove.append(patternIncluster)
                candidateRepPattern = [x for x in candidateRepPattern if x not in haveRePattern_remove]
        # 去除掉存在两个子集的代表性模式
        haveRePattern_remove = []
        for rePattern in representativePattern:
            patternName_list = rePattern.split()
            subsets = [subset for r in range(len(patternName_list) - 1, len(patternName_list)) for subset in
                       combinations(patternName_list, r)]
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
        minePatternRegion(representativePattern)  # 挖掘代表性模式的频繁区域
        haveGetAllrepattern_flag = 1
        for rePattern in representativePattern:
            if rePattern not in patterns_region_rowDict:
                haveGetAllrepattern_flag = 0
                cluster_patternDict[key].remove(rePattern)
        if haveGetAllrepattern_flag == 1:
            representativePattern_dict[key] = representativePattern

end2_time = time.time()
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
#         if patternName not in allPattern:
#             print(patternName)
