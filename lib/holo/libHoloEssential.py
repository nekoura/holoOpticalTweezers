import cupy as cp
from skimage.metrics import structural_similarity


class Holo:
    def __init__(self, targetImg: cp.ndarray, maxIterNum: int, **kwargs):
        """
        全息图生成 相关算法

        :param targetImg: 归一化目标图像
        :param maxIterNum: 最大迭代次数
        :keyword initPhase: 初始相位 (mode, phase) type=tuple(int, cp.ndarray)
        :keyword iterTarget: 迭代目标 (mode, val) type=tuple(int, float)
        :keyword uniList: 均匀性记录 type=list
        :keyword effiList: 光场效率记录 type=list
        :keyword RMSEList: 均方根误差记录 type=list
        """
        self.targetImg = targetImg
        self.maxIterNum = maxIterNum
        self.initPhase = kwargs.get('initPhase', (0, None))
        self.iterTarget = kwargs.get('iterTarget', (0, 0.95))
        self.normalizedAmp = None
        self.uniList = kwargs.get('uniList', [])
        self.effiList = kwargs.get('effiList', [])
        self.RMSEList = kwargs.get('RMSEList', [])
        self.SSIMList = kwargs.get('SSIMList', [])

        self.signalRegion = self.targetImg > 0
        self.nonSigRegion = self.targetImg == 0

    def uniformityCalc(self):
        """
        均匀性评价
        """
        maxI = cp.max(self.normalizedAmp[self.targetImg == 1])
        minI = cp.min(self.normalizedAmp[self.targetImg == 1])

        uniformity = 1 - (maxI - minI) / (maxI + minI)

        self.uniList.append(float(uniformity))

    def efficiencyCalc(self):
        """
        光场利用率评价
        """
        currentA = cp.sum(self.normalizedAmp[self.targetImg > 0])
        targetA = cp.sum(self.targetImg[self.targetImg > 0])

        efficiency = currentA / targetA

        self.effiList.append(float(efficiency))

    def RMSECalc(self):
        """
        均方根误差评价
        """
        retrievedI = cp.abs(self.normalizedAmp) ** 2
        targetI = cp.abs(self.targetImg) ** 2
        RMSE = cp.sqrt(
            cp.sum(retrievedI - targetI) ** 2 / cp.sum(targetI) ** 2
        )
        self.RMSEList.append(float(RMSE))

    def SSIMCalc(self):
        SSIM = structural_similarity(
            cp.asnumpy(self.targetImg),
            cp.asnumpy(self.normalizedAmp),
            data_range=cp.asnumpy(self.targetImg).max() - cp.asnumpy(self.targetImg).min()
        )

        self.SSIMList.append(SSIM)

    def iterAnalyze(self) -> bool:
        """
        迭代指标评价

        todo: SSIM PSNR

        :return: 是否终止迭代
        """
        # 检查相位恢复结果的均匀度
        self.uniformityCalc()
        # 检查相位恢复结果的光能利用率
        self.efficiencyCalc()
        # 检查相位恢复结果的RMSE
        self.RMSECalc()
        # 检查相位恢复结果的SSIM
        self.SSIMCalc()

        if self.iterTarget[0] == 0:
            # RMSE小于等于设置阈值
            if self.RMSEList[-1] <= self.iterTarget[1]:
                return True
        elif self.iterTarget[0] == 1:
            # SSIM大于等于设置阈值
            if self.SSIMList[-1] >= self.iterTarget[1]:
                return True
        elif self.iterTarget[0] == 2:
            # 光能利用率大于等于设置阈值
            if self.effiList[-1] >= self.iterTarget[1]:
                return True
        elif self.iterTarget[0] == 3:
            # 均匀度大于等于设置阈值
            if self.uniList[-1] >= self.iterTarget[1]:
                return True

    def phaseInitialization(self) -> cp.array:
        """
        相位初始化

        :return: 初始迭代相位
        """
        if self.initPhase[0] == 1:
            # 以目标光场IFFT作为初始迭代相位以增强均匀性 v2
            phase = cp.fft.ifftshift(cp.fft.ifft2(self.targetImg))
        elif self.initPhase[0] == 2:
            # 自定义相位 todo:带限初始相位
            phase = self.initPhase[1]
        else:
            # 以随机相位分布作为初始迭代相位 (默认）
            phase = cp.random.rand(self.targetImg.shape[0], self.targetImg.shape[1])

        return phase

    @staticmethod
    def normalize(img: cp.ndarray):
        """
        归一化

        :param img: 输入图像
        :return: 归一化图像
        :rtype: cp.ndarray
        """
        return (img - cp.min(img)) / (cp.max(img) - cp.min(img))

    @staticmethod
    def genHologram(phase: cp.ndarray):
        """
        自相位生成全息图

        :param phase: 输入相位
        :return: 全息图
        """
        # 相位校正，相位为负+2pi，为正保持原样
        phase = cp.where(phase < 0, phase + 2 * cp.pi, phase)
        holoImg = Holo.normalize(phase) * 255

        return holoImg.astype("uint8")

    @staticmethod
    def reconstruct(holoU: cp.ndarray, d: float, wavelength: float):
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
