import cupy as cp


def uniformityCalc(normIntensity, target) -> float:
    """
    均匀性评价

    :param normIntensity: 归一化光场强度
    :param target: 目标图像
    :return 均匀性
    :rtype: float
    """
    maxI = cp.max(normIntensity[target == 1])
    minI = cp.min(normIntensity[target == 1])

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


def RMSECalc(normIntensity, target) -> float:
    RMSE = cp.sqrt(
        cp.sum(cp.abs(normIntensity) ** 2 - cp.abs(target) ** 2) ** 2 / cp.sum(cp.abs(target) ** 2) ** 2
    )
    return float(RMSE)


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
    return cp.asnumpy(H * uFFT)
