# Origin Author: TOMOYUKI KUROSAWA (https://github.com/kurokuman/Gerchberg-Saxton-algorithm)
# See https://doi.org/10.1364/OE.15.001913

import cupy as cp


def GSiteration(maxIterNum: int, effThres: float, targetImg, uniList: list, effiList: list) -> tuple:
    """
    GS迭代算法

    :param maxIterNum: 最大迭代次数
    :param effThres: 迭代目标（均匀性）
    :param targetImg: 目标图像
    :param uniList: 均匀性记录
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

    # 光场复振幅：生成和target相同尺寸的空数组，必须是复数
    u = cp.empty_like(targetImg, dtype="complex")

    # 初始化光场
    uTarget = targetImg
    # 初始化权重数组
    uWeighted = cp.empty_like(targetImg)
    normIntensity = cp.empty_like(targetImg)

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
        uniList.append(uniformity)

        # 检查生成光场的光能利用率
        efficiency = efficiencyCalc(normIntensity, targetImg)
        effiList.append(efficiency)

        uWeighted = addWeight(uWeighted, targetImg, uTarget, normIntensity)
        uWeighted = normalize(uWeighted)
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

    return u, phase, normIntensity


def addWeight(uWeighted, target, uTarget, normIntensity):
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


def uniformityCalc(intensity, target) -> float:
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


def efficiencyCalc(normIntensity, target) -> float:
    """
    光场利用率评价

    :param normIntensity: 归一化光场强度
    :param target: 目标图像
    :return 光场利用率
    :rtype: float
    """
    efficiency = cp.sum(normIntensity[target == 1]) / cp.sum(target[target == 1])

    return float(efficiency)


def normalize(img):
    """
    归一化

    :param img: 输入图像
    :return: 归一化图像
    """
    return (img - cp.min(img)) / (cp.max(img) - cp.min(img))


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


def reconstruct(holoU, d, wavelength):
    """
    重建光场还原

    :param holoU: 全息图光场
    :param d: 衍射距离
    :param wavelength: 光源波长
    :return: 重建光场
    """
    k = 2 * cp.pi / wavelength
    H = cp.exp(1j * (k * d / wavelength))
    uFFT = cp.fft.fftshift(
        cp.fft.fft2(
            cp.fft.fftshift(
                cp.asarray(holoU)
            )
        )
    )
    uAbs = cp.abs(H * uFFT)
    uReconstruct = normalize(uAbs) * 255
    return cp.asnumpy(uReconstruct)
