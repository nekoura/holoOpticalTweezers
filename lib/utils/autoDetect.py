import sys
import cv2
import math
import numpy as np
import cupy as cp
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from multiprocessing import Process
from lib.holo.libHoloAlgmGPU import WCIA
from lib.utils.utils import Utils

from PyQt6.QtCore import QTimer
import time

class FeaturesDetect:
    @staticmethod
    def cutImg(image, center):
        # 计算圆形中心到图片边界的最小距离
        minDist = min(center[0], center[1], image.shape[1] - center[0] - 1, image.shape[0] - center[1] - 1)

        # 裁剪边长为最小距离的两倍
        squareSize = minDist * 2

        # 计算裁剪区域的起始点和结束点
        startPoint = (max(0, center[0] - squareSize // 2), max(0, center[1] - squareSize // 2))
        endPoint = (
            min(image.shape[1], center[0] + squareSize // 2), min(image.shape[0], center[1] + squareSize // 2))

        # 裁剪图片
        croppedImg = image[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]

        # 调整图片大小为正方形
        squareImg = cv2.resize(croppedImg, (squareSize, squareSize))

        # 确保图片是正方形的
        height, width = squareImg.shape[:2]

        # 缩放图片到610x610
        resizedImg = cv2.resize(squareImg, (610, 610), interpolation=cv2.INTER_AREA)

        # 创建1080x1080的空白图像
        comboImg = np.zeros((1080, 1080), dtype=np.uint8)

        # 确定缩放后的图片在空白图像中的位置
        xOffset = (1080 - 610) // 2
        yOffset = (1080 - 610) // 2

        # 将缩放后的图片粘贴到空白图像的中心
        comboImg[yOffset:yOffset + 610, xOffset:xOffset + 610] = resizedImg

        return comboImg

    @staticmethod
    def detectCircles(image) -> np.ndarray:
        # 应用高斯模糊减少图像噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 检测圆形，这里参数可能需要根据实际情况调整
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=13, minRadius=5, maxRadius=20
        )

        # 将检测到的圆形转换为列表形式的坐标
        circles = np.round(circles[0, :]).astype("int")

        return circles[:, :2]

    @staticmethod
    def drawMatches(currImg, targetImg, matches):
        # 创建一个新图像，用于绘制连线
        bgImage = cv2.addWeighted(currImg, 0.5, targetImg, 0.5, 0)

        # 遍历匹配的点对
        for (currentPoint, targetPoint) in matches:

            # 确保current_point和target_point是长度为2的元组
            currentLoc = tuple(currentPoint[:2])  # 取前两个元素作为x, y坐标
            targetLoc = tuple(targetPoint[:2])  # 同上

            # 绘制连线
            cv2.line(bgImage, currentLoc, targetLoc, (255, 255, 0), 1)

        return bgImage

    @staticmethod
    def match(currentPts, targetPts, cDist, cAngle):
        # 计算所有点对之间的距离
        distMatrix = distance.cdist(targetPts, currentPts, 'euclidean')

        # 初始化匹配列表
        matches = []

        # 初始化一个集合，用于存储已匹配的当前点索引
        matched_current = set()

        # 计算目标点之间的法线向量
        targetNormalVecs = []
        for i in range(len(targetPts)):
            if i == len(targetPts) - 1:  # 最后一个点
                nextIdx = i - 1  # 使用第一个点作为下一个点
            else:
                nextIdx = i + 1
            targetVec = np.array(targetPts[nextIdx]) - np.array(targetPts[i])
            if np.linalg.norm(targetVec) > 0:  # 确保不是零向量
                targetNormalVecs.append(np.array([-targetVec[1], targetVec[0]]))
            else:
                targetNormalVecs.append([0, 0])  # 如果是零向量，使用零向量作为法线

        # 遍历每个目标点
        for i, targetPt in enumerate(targetPts):
            # 为当前目标点找到最近的当前点
            minScore = float('inf')
            minIndex = -1
            for j, currentPt in enumerate(currentPts):
                if j not in matched_current:
                    dist = distMatrix[i][j]
                    currTargetVec = np.array(currentPt) - np.array(targetPt)
                    # 计算两个向量之间的夹角的余弦值
                    cosAngle = (np.dot(currTargetVec, targetNormalVecs[i]) /
                                (np.linalg.norm(currTargetVec) * np.linalg.norm(targetNormalVecs[i])))
                    if cosAngle > 1 or cosAngle < -1:
                        # 确保cos_angle在-1到1之间
                        cosAngle = np.abs(np.clip(cosAngle, -1, 1))
                    # 使用距离和余弦值作为加权因素
                    # 夹角越大，cos_angle越小，权重越小
                    score = cDist * dist + cAngle * dist * (1 - cosAngle)
                    if score < minScore:
                        minScore = score
                        minIndex = j

            # 如果找到了最近的点，则添加到匹配列表
            if minIndex != -1:
                matches.append((targetPt, currentPts[minIndex]))
                matched_current.add(minIndex)

        return matches


class FeaturesSort:
    def __init__(self, targetPts, thres):
        self.maxindexA = 0
        self.maxindexB = 0
        self.targetPts = targetPts
        self.totalOrderA = np.zeros_like(targetPts)
        self.totalOrderB = np.zeros_like(targetPts)
        self.thres = thres
        self.clusterNum = 1

    def detectCluster(self):
        kmeans = KMeans(n_clusters=self.clusterNum)
        # 将点转换为numpy数组
        target_points_array = np.array(self.targetPts)
        # 执行聚类 获取每个点的簇标签
        kmeans.fit(target_points_array)

        # 聚类排序
        # 根据簇标签对点进行分组
        cluster = {i: [] for i in range(self.clusterNum)}
        for pt, label in zip(target_points_array, kmeans.labels_):
            cluster[label].append(tuple(pt))

        # 对每个簇内的点按照y坐标排序
        for label, pts in cluster.items():
            cluster[label].sort(key=lambda x: x[1])

        return cluster

    @staticmethod
    def findClosestPoints(points) -> dict:
        """
        目标点集中，每个点和余下其他点的距离排序

        :param list points: 目标点集合
        :return: 点的距离集合
        """
        distances = defaultdict(list)
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j:
                    # 计算两点之间的欧几里得距离
                    dist = Utils.calcEucDist(p1, p2)
                    distances[p1].append((dist, p2))
        for k in points:
            # 根据距离排序
            distances[k].sort()
        return distances

    # 第一次全连接点
    def connectPoints(self, points) -> dict:
        """
        对点进行初步连接

        :param list points: 目标点集合
        :return: 连接结果
        """
        connections = defaultdict(list)
        for point, closest in self.findClosestPoints(points).items():
            if len(closest) >= 2:
                connections[point].extend([closest[0][1], closest[1][1]])
        return connections

    @staticmethod
    def checkConnections(connections, points, thres):
        """
        检查连接正确性

        :param dict connections: 点的连接
        :param list points: 目标点集合
        :param thres: 距离差异阈值
        """
        # 计算所有点对之间的平均距离
        dist = [Utils.calcEucDist(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
        if dist:
            avgDist = sum(dist) / len(dist)
        else:
            return  # 如果没有点对，则直接返回

        # 对每个点进行处理
        for point in points:
            # 如果当前点没有连接，则尝试添加一个连接
            if not connections[tuple(point)]:
                closestPt = min(points, key=lambda p: Utils.calcEucDist(p, point))
                connections[tuple(point)].append(closestPt)

            # 如果当前点有两个连接，检查它们之间的距离是否相近
            if len(connections[tuple(point)]) == 2:
                twoConnDist = sorted(
                    [Utils.calcEucDist(point, connPt) for connPt in connections[tuple(point)]]
                )
                # 如果两个连接的距离相差超过平均距离的某个倍数，删除较大的连接
                if twoConnDist[1] - twoConnDist[0] > avgDist * thres:
                    connections[tuple(point)].pop()

        # 检查是否有需要重新添加连接的情况
        for point, conn in connections.items():
            if len(conn) == 0:
                # 重新添加连接，选择最近的点
                closest = sorted(points, key=lambda p: Utils.calcEucDist(p, point))
                connections[point] = [closest[1]]


    @staticmethod
    def sortCluster(connections, points):
        """
        聚类排序结果输出

        :param dict connections: 连接结果
        :param list points:
        :return: retCode，起始点，排列顺序
        """
        # 根据聚类ID获取该聚类的连接信息
        connect = {tuple(point): conn for point, conn in connections.items() if tuple(point) in {tuple(p) for p in points}}

        # 找到只有一个连接的点作为起点
        startPoints = [point for point, connections in connect.items() if len(connections) == 1]

        # 找不到则直接返回
        if not startPoints:
            print("No starting point found with exactly one connection.")
            return -1, None, None

        # 对每个只有一个连接的点，执行一次广度优先搜索
        linkOrders = []
        for startPoint in startPoints:
            visited = set()
            linkOrder = []
            que = deque([startPoint])

            while que:
                currentPoint = que.popleft()
                if currentPoint not in visited:
                    visited.add(currentPoint)
                    linkOrder.append(currentPoint)
                    # 将当前点的所有连接点加入队列
                    for connPoint in connect.get(currentPoint, []):
                        que.append(connPoint)

            # 将输出顺序中的点元组转换回原始点的格式
            linkOrder = [tuple(point) for point in linkOrder]
            linkOrders.append(linkOrder)

        return 0, startPoints, linkOrders

    def calc(self):
        """
        计算最优连接

        :return: 顺序A，顺序B
        """
        cluster = self.detectCluster()
        # 寻找每个簇的最优连接
        connnected = defaultdict(dict)
        for cid, group in cluster.items():
            # 连接点
            connnected[cid] = self.connectPoints(group)

        for cid, ptConnections in connnected.items():
            # 检查连接规则
            for group in cluster.values():
                self.checkConnections(ptConnections, group, self.thres)

            # 输出排序后点
            ret, startPts, orders = self.sortCluster(connnected[cid], cluster[cid])

            for index, order in enumerate(orders[0]):
                self.totalOrderA[self.maxindexA] = order
                self.maxindexA += 1

            for index, order in enumerate(orders[1]):
                self.totalOrderB[self.maxindexB] = order
                self.maxindexB += 1

            return self.totalOrderA, self.totalOrderB


class FrameGeneratorWorker(Process):
    def __init__(self, matchedPairs, framePipeSender):
        Process.__init__(self)
        self.matchedPairs = matchedPairs
        self.framePipeSender = framePipeSender
        # 初始化未启动点列表和已结束点列表
        self.unstartedPoints = [start for _, start in self.matchedPairs]
        self.endedPoints = []
        # 当前总帧数
        self.currentFrame = 0
        self.currentPoint = 0
        # 预设的步长
        self.stepLength = 5
        self.width = 1080
        self.height = 1080

    def calcSteps(self, dist):
        return max(1, int(math.ceil(dist / self.stepLength)))

    def run(self):
        while self.currentPoint < len(self.matchedPairs):
            end, start = self.matchedPairs[self.currentPoint]
            # 从未启动点列表中移除当前点
            self.unstartedPoints.remove(start)

            # 计算步数
            dist = Utils.calcEucDist(start, end)
            steps = self.calcSteps(dist)

            # 逐帧绘制并保存
            for i in range(steps + 1):  # 包括起始点和结束点
                # 创建新的黑色图像
                frame = np.zeros((self.height, self.width), dtype=np.uint8)

                # 绘制所有未启动的起点
                for point in self.unstartedPoints:
                    cv2.circle(frame, point, 5, (255, 255, 255), -1)

                # 绘制所有已结束的终点
                for point in self.endedPoints:
                    cv2.circle(frame, point, 5, (255, 255, 255), -1)

                # 如果是起始帧，绘制起始点
                if i == 0:
                    cv2.circle(frame, start, 5, (255, 255, 255), -1)

                # 计算并绘制当前点的移动位置
                if i > 0:
                    x = int(start[0] + (end[0] - start[0]) * (i / steps))
                    y = int(start[1] + (end[1] - start[1]) * (i / steps))
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

                # 保存当前帧
                self.framePipeSender.send((cp.asarray(frame / 255), self.currentFrame))
                self.currentFrame += 1

            # 移动结束，将终点添加到已结束点列表
            self.endedPoints.append(end)
            self.currentPoint += 1
        else:
            self.framePipeSender.close()
            sys.exit(0)


class HoloGeneratorWorker(Process):
    def __init__(self, framePipeReceiver, holoPipeSender, maxIterNum=40, iterTarget=0.01):
        Process.__init__(self)
        self.framePipeReceiver = framePipeReceiver
        self.holoPipeSender = holoPipeSender
        self.maxIterNum = maxIterNum
        self.iterTarget = iterTarget

    def run(self):
        while True:
            try:
                (frame, index) = self.framePipeReceiver.recv()
            except EOFError:
                self.framePipeReceiver.close()
                self.holoPipeSender.close()
                sys.exit(0)
            else:
                u, phase = WCIA.staticIterate(
                    frame,
                    self.maxIterNum,
                    initPhase=(1, None),
                    iterTarget=(0, self.iterTarget)
                )
                self.holoPipeSender.send((phase, index))
