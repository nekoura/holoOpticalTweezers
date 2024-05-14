import cupy as cp
import torch
import torch.nn.functional as F
import math


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
        self.SSIMList = kwargs.get('RMSEList', [])

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

    def SSIMCalc(self, window_size=11, window=None, size_average=True, full=False):
        def gaussian(window_size, sigma):
            """
            Generates a list of Tensor values drawn from a gaussian distribution with standard
            diviation = sigma and sum of all elements = 1.

            Length of list = window_size
            """
            gauss = torch.Tensor(
                [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
                )
            return gauss / gauss.sum()

        def create_window(window_size, channel=1):

            # Generate an 1D tensor containing values sampled from a gaussian distribution
            _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

            # Converting to 2D
            _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

            window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

            return window

        pad = window_size // 2

        [height, width, channels] = self.normalizedAmp.shape

        # if window is not provided, init one
        if window is None:
            real_size = min(window_size, height, width)  # window should be atleast 11x11
            window = create_window(real_size, channel=channels).to(self.normalizedAmp.device)

        # calculating the mu parameter (locally) for both images using a gaussian filter
        # calculates the luminosity params
        mu1 = F.conv2d(self.normalizedAmp, window, padding=pad, groups=channels)
        mu2 = F.conv2d(self.targetImg, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component
        sigma1_sq = F.conv2d(
            self.normalizedAmp * self.normalizedAmp, window, padding=pad, groups=channels
        ) - mu1_sq
        sigma2_sq = F.conv2d(
            self.targetImg * self.targetImg, window, padding=pad, groups=channels
        ) - mu2_sq
        sigma12 = F.conv2d(
            self.normalizedAmp * self.targetImg, window, padding=pad, groups=channels
        ) - mu12

        # Some constants for stability
        C1 = 0.01 ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = 0.03 ** 2

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            ret = ssim_score.mean()
        else:
            ret = ssim_score.mean(1).mean(1).mean(1)

        if full:
            return ret, contrast_metric

        self.SSIMList.append(float(ret))

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
            if self.SSIMList[-1] <= self.iterTarget[1]:
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
