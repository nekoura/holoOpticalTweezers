# Ref https://doi.org/10.3390/app10103652

import cupy as cp
from lib.holo.libHolo_GPU import Holo


class WCIA(Holo):
    def __init__(self, targetImg: cp.ndarray, maxIterNum: int, **kwargs):
        """
        初始化变量 (See Fig.3, Eq.1 and Eq.6)

        目标图像 A_target = self.Atarget
        相位 φ = self.phase
        迭代k 图像平面（输入）复振幅  A_k = self.Ak
        迭代k 全息平面复振幅  a_k = self.ak
        迭代k 相位提取和振幅正则化后复振幅  a_k' = self.aK
        迭代k 图像平面（重建）复振幅分布  A_k' = self.AK
        自适应约束参数  β_k = self.bk
        全息平面强制振幅约束  A_holo = self.Aholo
        图像平面强制振幅约束  A_con = self.Acon
        """
        super().__init__(targetImg, maxIterNum, **kwargs)

        self.Atarget = self.targetImg
        self.phase = self.phaseInitialization()

        self.Ak = self.Atarget * cp.exp(1j * self.phase)
        self.ak = cp.empty_like(self.targetImg, dtype="complex")
        self.aK = cp.empty_like(self.targetImg, dtype="complex")
        self.AK = cp.empty_like(self.targetImg, dtype="complex")

        self.bk = 1e-8
        Eholo = 1
        H = self.targetImg.shape[0] * self.targetImg.shape[1]
        self.Aholo = cp.sqrt(Eholo / H)
        self.Acon = cp.empty_like(self.Atarget, dtype="complex")

    def iteration(self) -> tuple:
        """
        WCIA迭代算法
        """
        for n in range(self.maxIterNum):
            self.ak = cp.fft.ifft2(cp.fft.ifftshift(self.Ak))

            self.aK = self.Aholo * (self.ak / cp.abs(self.ak))

            self.AK = cp.fft.fftshift(cp.fft.fft2(self.aK))

            # 向像平面光场添加强制振幅约束(See Eq.1)
            self.Acon[self.signalRegion] = (
                cp.abs(self.Ak[self.signalRegion]) *
                (cp.abs(self.Atarget[self.signalRegion]) / cp.abs(self.AK[self.signalRegion])) ** self.bk
            )
            self.Acon[self.nonSigRegion] = cp.abs(self.Ak[self.nonSigRegion])

            self.Ak = self.Acon * (self.AK / cp.abs(self.AK))
            self.bk = cp.sqrt(self.bk)

            self.phase = cp.angle(self.aK)

            # I=|u|^2
            # intensity = cp.abs(AK) ** 2
            # 归一化光强
            self.normalizedA = self.normalize(cp.abs(self.AK))

            if self.iterAnalyze():
                break

        # 显存GC
        cp._default_memory_pool.free_all_blocks()

        return self.aK, self.phase


