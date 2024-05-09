# Origin Author: TOMOYUKI KUROSAWA (https://github.com/kurokuman/Gerchberg-Saxton-algorithm)
# Ref https://doi.org/10.1364/OE.15.001913

import cupy as cp
from lib.holo.libHolo_GPU import Holo


class GSW(Holo):
    def __init__(self, targetImg: cp.ndarray, maxIterNum: int, **kwargs):
        """
        初始化变量

        目标图像 A_target = self.Atarget
        相位 φ = self.phase
        迭代k 图像平面（输入）复振幅  A_k = self.Ak
        迭代k 全息平面复振幅  a_k = self.ak
        迭代k 相位提取和振幅正则化后复振幅  a_k' = self.aK
        迭代k 图像平面（重建）复振幅分布  A_k' = self.AK
        加权后振幅  A_w = self.Aw
        """
        super().__init__(targetImg, maxIterNum, **kwargs)

        # 初始化光场 权重数组
        self.Atarget = self.targetImg
        self.phase = self.phaseInitialization()
        self.Aw = cp.empty_like(self.targetImg)

        self.Ak = self.Atarget * cp.exp(1j * self.phase)
        self.ak = cp.empty_like(self.targetImg, dtype="complex")
        self.aK = cp.empty_like(self.targetImg, dtype="complex")
        self.AK = cp.empty_like(self.targetImg, dtype="complex")

    def iteration(self) -> tuple:
        """
        GS迭代算法

        todo: debug
        """

        for n in range(self.maxIterNum):
            self.ak = cp.fft.ifft2(cp.fft.ifftshift(self.Ak))

            self.aK = 1 * cp.exp(1j * cp.angle(self.ak))

            self.AK = cp.fft.fft2(cp.fft.fftshift(self.aK))

            self.normalizedA = Holo.normalize(cp.abs(self.AK))

            # 向迭代光场添加权重(See Eq.19)
            self.Aw[self.signalRegion] = Holo.normalize(
                self.Atarget[self.signalRegion] *
                ((self.targetImg[self.signalRegion] / self.normalizedA[self.signalRegion]) ** 0.5)
            )

            self.Ak = self.Atarget * self.Aw * cp.exp(1j * cp.angle(self.AK))

            self.phase = cp.angle(self.aK)

            if self.iterAnalyze():
                break

        # 显存GC
        cp._default_memory_pool.free_all_blocks()

        return self.aK, self.phase

