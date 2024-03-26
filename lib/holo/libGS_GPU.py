# Origin Author: TOMOYUKI KUROSAWA (https://github.com/kurokuman/Gerchberg-Saxton-algorithm)
# See https://doi.org/10.1364/OE.15.001913

import cupy as cp


def addWeight(weightedU, target, targetU, normIntensity):
    """
    向迭代光场添加权重(See Eq.19)

    :param weightedU: 迭代前(step k-1)加权光场强度
    :param target: 目标图像
    :param targetU: 目标光场
    :param normIntensity: 归一化光强
    :return: 迭代后(step k)加权光场强度
    """
    weightedU[target == 1] = ((target[target == 1] / normIntensity[target == 1]) ** 0.5) * targetU[target == 1]
    return weightedU


def uniformityCalc(intensity, target):
    """
    均匀性评价，达到一定值后终止迭代(See Eq.4-2)

    :param intensity: 输入光场强度
    :param target: 目标图像
    :return 均匀性
    :rtype: float
    """
    intensity = intensity / cp.max(intensity)

    maxI = cp.max(intensity[target == 1])
    minI = cp.min(intensity[target == 1])

    uniformity = float(1 - (maxI - minI) / (maxI + minI))

    return uniformity


def GSiteration(maxIterNum: int, uniThres: float, targetImg, unfmList: list):
    """
    GS迭代算法

    :param int maxIterNum: 最大迭代次数
    :param float uniThres: 迭代目标（均匀性）
    :param targetImg: 目标图像
    :param list unfmList: 均匀性记录
    :return: 相位, 归一化光强
    :rtype: tuple
    """
    # (deprecated)初始迭代相位：以随机相位分布作为初始迭代相位
    # phase = cp.random.rand(height, width)

    # (deprecated)初始迭代相位：以目标光场IFFT作为初始迭代相位以增强均匀性
    # height, width = targetImg.shape[:2]
    # initU = cp.fft.ifftshift(cp.fft.ifft2(targetImg))
    # phase = cp.angle(initU) + 2*cp.pi*(cp.random.uniform(0,1,(height, width))-0.5)/cp.sinc(cp.abs(initU)))

    # 初始迭代相位：以目标光场IFFT作为初始迭代相位以增强均匀性 v2
    # todo:高斯面型
    phase = cp.fft.ifftshift(cp.fft.ifft2(targetImg))

    # 光场复振幅：生成和target相同尺寸的空数组，必须是负数
    u = cp.empty_like(targetImg, dtype="complex")

    # 初始化光场
    targetU = targetImg
    # 初始化权重数组
    weightedU = cp.empty_like(targetImg)

    for n in range(maxIterNum):
        # 输入到LCOS上的复振幅光场，设入射LCOS的初始光强相对值为1，N=1为随机相位，N>1为迭代相位
        u.real = cp.cos(phase)
        u.imag = cp.sin(phase)

        # 模拟透镜传递函数（向光阱正向传播）
        u = cp.fft.fft2(u)
        u = cp.fft.fftshift(u)
        # ---------------------------------

        phase = cp.angle(u)

        # I=|u|^2
        intensity = cp.abs(u) ** 2
        # 归一化光强
        normIntensity = normalize(intensity)

        # 检查生成光场的均匀度
        uniformity = uniformityCalc(intensity, targetImg)
        unfmList.append(uniformity)

        weightedU = addWeight(weightedU, targetImg, targetU, normIntensity)
        weightedU = normalize(weightedU)
        targetU = weightedU

        u.real = weightedU * cp.cos(phase)
        u.imag = weightedU * cp.sin(phase)

        # 模拟透镜传递函数（向LCOS反向传播）
        u = cp.fft.ifftshift(u)
        u = cp.fft.ifft2(u)
        # ---------------------------------

        phase = cp.angle(u)

        # if uniformity >= uniThres:
        #     break

    return (phase, normIntensity)


# 归一化
def normalize(img):
    maxI = cp.max(img)
    minI = cp.min(img)

    result = ((img - minI) / (maxI - minI))

    return result


# 生成全息图
def genHologram(phase):
    """
    自相位生成全息图

    :param phase: 输入相位
    :return: 全息图
    """
    # 相位校正，相位为负+2pi，为正保持原样
    phase = cp.where(phase < 0, phase + 2 * cp.pi, phase)
    holo = normalize(phase) * 255
    holo = holo.astype("uint8")
    return holo


