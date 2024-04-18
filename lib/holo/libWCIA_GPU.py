# See https://doi.org/10.3390/app10103652

import cupy as cp
from lib.holo import libHolo_GPU as Holo


def WCIAiteration(targetImg: cp.ndarray, maxIterNum: int, **kwargs) -> tuple:
    """
    WCIA迭代算法

    :param targetImg: 目标图像
    :param maxIterNum: 最大迭代次数
    :keyword initPhase: 初始相位 (mode, phase) type=tuple(int, cp.ndarray)
    :keyword iterTarget: 迭代目标 (mode, val) type=tuple(int, float)
    :keyword uniList: 均匀性记录 type=list
    :keyword effiList: 光场效率记录 type=list
    :keyword RMSEList: 均方根误差记录 type=list
    :return: 光场，相位
    :rtype: tuple(cupy.ndarray, cupy.ndarray)
    """

    initPhase = kwargs.get('initPhase', (0, None))
    iterTarget = kwargs.get('iterTarget', (0, 0.95))
    uniList = kwargs.get('uniList', [])
    effiList = kwargs.get('effiList', [])
    RMSEList = kwargs.get('RMSEList', [])

    if initPhase[0] == 0:
        # 初始迭代相位：以随机相位分布作为初始迭代相位
        phase = cp.random.rand(targetImg.shape[0], targetImg.shape[1])
    elif initPhase[0] == 1:
        # 初始迭代相位：以目标光场IFFT作为初始迭代相位以增强均匀性 v2
        phase = cp.fft.ifftshift(cp.fft.ifft2(targetImg))
    elif initPhase[0] == 2:
        # 初始迭代相位：自定义相位 todo:高斯面型
        phase = initPhase[1]

    # 初始化光场
    Atarget = targetImg

    Ak = Atarget * cp.exp(1j * phase)

    Etarget = cp.abs(Atarget) ** 2
    H = targetImg.shape[0] * targetImg.shape[1]
    Aholo = cp.sqrt(1 / H)

    bk = 1e-8

    for n in range(maxIterNum):
        ak = cp.fft.ifft2(cp.fft.ifftshift(Ak))

        aK = Aholo * (ak / cp.abs(ak))

        AK = cp.fft.fftshift(cp.fft.fft2(aK))

        Acon = WCIAcalAcon(Ak, AK, Atarget, bk)

        Ak = Acon * (AK / cp.abs(AK))
        bk = cp.sqrt(bk)

        phase = cp.angle(aK)

        # I=|u|^2
        intensity = cp.abs(AK) ** 2
        # 归一化光强
        normIntensity = Holo.normalize(intensity)

        # 检查生成光场的均匀度
        uniformity = Holo.uniformityCalc(intensity, targetImg)
        uniList.append(uniformity)

        # 检查生成光场的光能利用率
        efficiency = Holo.efficiencyCalc(normIntensity, targetImg)
        effiList.append(efficiency)

        RMSE = Holo.RMSECalc(normIntensity, targetImg)
        RMSEList.append(RMSE)

        if iterTarget[0] == 0:
            # 光能利用率大于设置阈值
            if efficiency >= iterTarget[1]:
                break
        if iterTarget[0] == 1:
            # 均匀度大于设置阈值
            if uniformity >= iterTarget[1]:
                break
        elif iterTarget[0] == 2:
            # RMSE小于设置阈值
            if RMSE <= iterTarget[1]:
                break

    # 显存GC
    cp._default_memory_pool.free_all_blocks()

    return aK, phase


def WCIAcalAcon(Ak: cp.ndarray, AK: cp.ndarray, Atarget: cp.ndarray, bk: float) -> cp.ndarray:
    """
    向像平面光场添加强制振幅约束(See Eq.1)

    :param Ak: 输入图像在迭代k中振幅分布 Ak
    :param AK: 重建图像在迭代k中的振幅分布 Ak'
    :param Atarget: 目标图像振幅分布
    :param bk: 自适应参数 βk
    :return: 迭代后(step k)加权光场强度
    """

    Acon = cp.empty_like(Atarget, dtype="complex")

    Acon[Atarget > 0] = cp.abs(Ak[Atarget > 0]) * (cp.abs(Atarget[Atarget > 0]) / cp.abs(AK[Atarget > 0])) ** bk
    Acon[Atarget == 0] = cp.abs(Ak[Atarget == 0])
    return Acon
