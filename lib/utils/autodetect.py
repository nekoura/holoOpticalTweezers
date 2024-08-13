import sys
import cv2
import math
import numpy as np
import cupy as cp
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir
from lib.holo.libHoloAlgmGPU import WCIA
from lib.holo.libHoloEssential import Holo
from lib.utils.utils import Utils, Worker, MessageQueue
import threading


class FeaturesDetect:
    @staticmethod
    def detectCircles(image):
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊减少图像噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 检测圆形，这里参数可能需要根据实际情况调整
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=13, minRadius=5, maxRadius=20
        )

        # 将检测到的圆形转换为列表形式的坐标
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return circles

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

        # 可视化聚类结果

        targetImageCopy = targetImage
        for label, pts in cluster.items():
            color = (255 * (label % 3), 255 * ((label // 3) % 2), 0)
            for pt in pts:
                cv2.circle(targetImageCopy, pt, 5, color, -1)

        # 保存可视化结果
        cv2.imwrite('Clustered_Circles.jpg', targetImageCopy)

        # 打印每个簇的点
        for label, points in cluster.items():
            print(f"Cluster {label}: {points}")

        return cluster

    @staticmethod
    # 找出每个点到其他点距离最短的前两个点
    def findClosestPoints(points):
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
    def connectPoints(self, points):
        connections = defaultdict(list)
        for point, closest in self.findClosestPoints(points).items():
            if len(closest) >= 2:
                connections[point].extend([closest[0][1], closest[1][1]])
        return connections

    @staticmethod
    # 检查连接正确性
    def checkConnections(connections, points, thres):
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
                    print(f"  remove connection {connections[tuple(point)]}")
                    connections[tuple(point)].pop()

        # 检查是否有需要重新添加连接的情况
        for point, conn in connections.items():
            if len(conn) == 0:
                # 重新添加连接，选择最近的点
                closest = sorted(points, key=lambda p: Utils.calcEucDist(p, point))
                connections[point] = [closest[1]]

    # 绘制点和连接线
    @staticmethod
    def plotConnections(connections, points):
        plt.figure(figsize=(10, 8))
        for point in points:
            # 绘制点
            plt.plot(point[0], point[1], 'bo')
            # 绘制连接线
            for connect in connections[tuple(point)]:
                plt.plot([point[0], connect[0]], [point[1], connect[1]], 'r-')
        plt.grid(False)
        ax = plt.gca()  # 获取到当前坐标轴信息
        ax.set_aspect(1)
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        ax.invert_yaxis()  # 反转Y坐标轴
        plt.show()

    # 按连接顺序输出聚类中的点
    @staticmethod
    def sortCluster(connections, points):
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
        cluster = self.detectCluster()
        # 寻找每个簇的最优连接
        connnected = defaultdict(dict)
        for cid, group in cluster.items():
            # 连接点
            connnected[cid] = self.connectPoints(group)

        for cid, ptConnections in connnected.items():
            # 检查连接规则
            print(f"Cluster {cid}:")
            for group in cluster.values():
                self.checkConnections(ptConnections, group, self.thres)

            # 输出排序后点
            ret, startPts, orders = self.sortCluster(connnected[cid], cluster[cid])

            # 打印所有起点的输出顺序
            for index, order in enumerate(orders):
                print(f"  Starting from point {startPts[index]}: {order}")

            for index, order in enumerate(orders[0]):
                self.totalOrderA[self.maxindexA] = order
                self.maxindexA += 1

            for index, order in enumerate(orders[1]):
                self.totalOrderB[self.maxindexB] = order
                self.maxindexB += 1

            return self.totalOrderA, self.totalOrderB


class GenerateFrames:
    def __init__(self, points):
        self.points = points
        # 初始化未启动点列表和已结束点列表
        self.unstartedPoints = [start for _, start in points]
        self.endedPoints = []
        # 当前总帧数
        self.currentFrame = 0
        # 预设的步长
        self.stepLength = 5
        self.width = 1080
        self.height = 1080

    # 根据距离和预设的步长计算步数
    def calcSteps(self, dist):
        return max(1, int(math.ceil(dist / self.stepLength)))

    def move(self):
        # 逐点绘制动画
        for end, start in self.points:
            # 从未启动点列表中移除当前点
            self.unstartedPoints.remove(start)

            # 计算步数
            dist = Utils.calcEucDist(start, end)
            steps = self.calcSteps(dist)

            # 逐帧绘制并保存
            for i in range(steps + 1):  # 包括起始点和结束点
                # 创建新的黑色图像
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

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
                cv2.imwrite(f'./new111/frame_{self.currentFrame:05d}.png', frame)
                self.currentFrame += 1

            # 移动结束，将终点添加到已结束点列表
            self.endedPoints.append(end)


class HoloIter:
    def __init__(self, dirPath, maxIterNum, iterTarget):
        self.instances = []
        self.targetImgs = []
        self.dirPath = dirPath
        self.maxIterNum = maxIterNum
        self.iterTarget = iterTarget
        self.curr = 0

    def __iter__(self):
        self.load()
        return self

    def __next__(self):
        targetNormalized = self.targetImgs[self.curr] / 255
        target = cp.asarray(targetNormalized)
        instance = WCIA(target, self.maxIterNum, initPhase=(1, None), iterTarget=(0, self.iterTarget))
        print(f'Processing Img {self.curr}')
        u, phase = instance.iterate()
        message = (u, phase)
        self.curr += 1
        message_queue.send(message)  # 发送信号

    def load(self):
        print("Loading images:")
        for img in QDir(self.dirPath).entryList(['*.png', '*.jpg', '*.jpeg']):
            imagePath = QDir(self.dirPath).filePath(img)
            targetImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if targetImg is None:
                print(f"Warning: Cannot load {imagePath}")
            else:
                self.targetImgs.append(targetImg)
                print(f'{imagePath} appended')


class GeneratorWorker(Worker):
    def __init__(self):
        self.currentFrame = 0
        self.dirPath = f"./new111/"
        self.tempPath = f"./new111"

    def process(self):
        iterator = iter(HoloIter(self.dirPath, 10, 0.1))
        while self.currentFrame < Utils.calcFileNum(self.tempPath):
            next(iterator)
            self.currentFrame += 1

        print('Generator terminated')


class ImgSaveWorker(Worker):
    def __init__(self):
        self.currentFrame = 0
        self.prefix = f"./new222/holo_"
        self.tempPath = f"./new111"

    def process(self):
        while self.currentFrame < Utils.calcFileNum(self.tempPath):
            result = message_queue.receive()  # 接收信号
            self.save(result)
            self.currentFrame += 1

        print('Saver terminated')

    def save(self, result):
        holoImg = cp.asnumpy(Holo.genHologram(result[1]))
        holoU = cp.asnumpy(result[0])
        cv2.imwrite(f'{self.prefix}{self.currentFrame:05d}.png', cv2.flip(holoImg, 0))
        # np.save(f"{self.prefix}{self.currentFrame:05d}.npy", holoU)
        print(f'Write "{self.prefix}{self.currentFrame:05d}.png"')


def main():
    # 检测特征点
    currentPoints = FeaturesDetect.detectCircles(currentImage)
    targetPoints = FeaturesDetect.detectCircles(targetImage)

    currentPoints = currentPoints[:, :2]
    targetPoints = targetPoints[:, :2]

    totalOrderA, _ = FeaturesSort(targetPoints, 0.1).calc()

    matchedPairs = FeaturesDetect.match(currentPoints, totalOrderA, 0.55, 2.2)

    # 打印匹配结果
    for target, current in matchedPairs:
        print(f"目标位置: {target} 当前位置: {current}")

    GenerateFrames(matchedPairs).move()

    # ==========================================

    # 创建Worker实例
    generator_worker = GeneratorWorker()
    imgSaver_worker = ImgSaveWorker()

    # 启动线程或进程
    thread_generator = threading.Thread(target=generator_worker.process)
    thread_imgSaver = threading.Thread(target=imgSaver_worker.process)

    thread_generator.start()
    thread_imgSaver.start()

    # 等待线程完成
    thread_generator.join()
    thread_imgSaver.join()

    app = QApplication(sys.argv)
    directory_path = './new222/'  # 替换为你的图片文件夹路径
    interval = 500  # 图片间隔时间，单位毫秒
    window = ImageLoader(directory_path, interval)
    window.resize(1080, 1080)  # 根据图片大小调整窗口尺寸
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # 初始化消息队列
    message_queue = MessageQueue()
    # 读取图像
    currentImage = cv2.imread('20240708182214-R.jpg')
    targetImage = cv2.imread('CC.bmp')
    main()
