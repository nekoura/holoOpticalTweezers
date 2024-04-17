# Origin Author: TOMOYUKI KUROSAWA (https://github.com/kurokuman/Gerchberg-Saxton-algorithm)
# See https://doi.org/10.1364/OE.15.001913

import cupy as cp
from lib.holo import libHolo_GPU as Holo


def GSiteration(maxIterNum: int, effThres: float, targetImg, uniList: list, effiList: list) -> tuple:
    """
    GS迭代算法

    :param maxIterNum: 最大迭代次数
    :param effThres: 迭代目标（均匀性）
    :param targetImg: 目标图像
    :param uniList: 均匀性记录
    :param effiList: 光场效率记录
    :return: 光场，相位
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

    # 光场复振幅：生成和target相同尺寸的空数组，必须是复数
    u = cp.empty_like(targetImg, dtype="complex")

    # 初始化光场
    uTarget = targetImg
    # 初始化权重数组
    uWeighted = cp.empty_like(targetImg)

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
        normIntensity = Holo.normalize(intensity)

        # 检查生成光场的均匀度
        uniformity = Holo.uniformityCalc(intensity, targetImg)
        uniList.append(uniformity)

        # 检查生成光场的光能利用率
        efficiency = Holo.efficiencyCalc(normIntensity, targetImg)
        effiList.append(efficiency)

        uWeighted = GSaddWeight(uWeighted, targetImg, uTarget, normIntensity)
        uWeighted = Holo.normalize(uWeighted)
        uTarget = uWeighted

        u.real = uWeighted * cp.cos(phase)
        u.imag = uWeighted * cp.sin(phase)

        # 模拟透镜传递函数（向LCOS反向传播）
        u = cp.fft.ifftshift(u)
        u = cp.fft.ifft2(u)
        # ---------------------------------

        phase = cp.angle(u)

        if efficiency >= effThres:
            break

    # 显存GC
    cp._default_memory_pool.free_all_blocks()

    return u, phase


def GSaddWeight(uWeighted, target, uTarget, normIntensity):
    """
    向迭代光场添加权重(See Eq.19)

    :param uWeighted: 迭代前(step k-1)加权光场强度
    :param target: 目标图像
    :param uTarget: 目标光场
    :param normIntensity: 归一化光强
    :return: 迭代后(step k)加权光场强度
    """
    uWeighted[target > 0] = ((target[target > 0] / normIntensity[target > 0]) ** 0.5) * uTarget[target > 0]
    return uWeighted
