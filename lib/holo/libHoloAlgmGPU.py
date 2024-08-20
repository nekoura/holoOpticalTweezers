import cupy as cp
from lib.holo.libHoloEssential import Holo

# Ref https://doi.org/10.3390/app10103652
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


    def iterate(self) -> tuple:
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

            # 归一化光强
            self.normalizedAmp = self.normalize(cp.abs(self.AK))

            if self.iterAnalyze():
                break

        # 显存GC
        cp._default_memory_pool.free_all_blocks()

        return self.aK, self.phase

    @staticmethod
    def staticIterate(targetImg: cp.ndarray, maxIterNum: int, **kwargs):
        """
        WCIA迭代算法（静态）
        """
        Atarget = targetImg
        signalRegion = targetImg > 0
        nonSigRegion = targetImg == 0
        initPhase = kwargs.get('initPhase', (0, None))
        iterTarget = kwargs.get('iterTarget', (0, 0.01))
        RMSEList = kwargs.get('RMSEList', [])

        if initPhase[0] == 1:
            # 以目标光场IFFT作为初始迭代相位以增强均匀性 v2
            phase = cp.fft.ifftshift(cp.fft.ifft2(targetImg))
        else:
            # 以随机相位分布作为初始迭代相位 (默认）
            phase = cp.random.rand(targetImg.shape[0], targetImg.shape[1])

        Ak = Atarget * cp.exp(1j * phase)
        ak = cp.empty_like(targetImg, dtype="complex")
        aK = cp.empty_like(targetImg, dtype="complex")
        AK = cp.empty_like(targetImg, dtype="complex")

        bk = 1e-8
        Eholo = 1
        H = targetImg.shape[0] * targetImg.shape[1]
        Aholo = cp.sqrt(Eholo / H)
        Acon = cp.empty_like(Atarget, dtype="complex")

        for n in range(maxIterNum):
            ak = cp.fft.ifft2(cp.fft.ifftshift(Ak))

            aK = Aholo * (ak / cp.abs(ak))

            AK = cp.fft.fftshift(cp.fft.fft2(aK))
            # 向像平面光场添加强制振幅约束(See Eq.1)
            Acon[signalRegion] = (
                cp.abs(Ak[signalRegion]) *
                (cp.abs(Atarget[signalRegion]) / cp.abs(AK[signalRegion])) ** bk
            )
            Acon[nonSigRegion] = cp.abs(Ak[nonSigRegion])
            Ak = Acon * (AK / cp.abs(AK))
            bk = cp.sqrt(bk)
            phase = cp.angle(aK)

            # 归一化光强
            normalizedAmp = (cp.abs(AK) - cp.min(cp.abs(AK))) / (cp.max(cp.abs(AK)) - cp.min(cp.abs(AK)))

            retrievedI = cp.abs(normalizedAmp) ** 2
            targetI = cp.abs(targetImg) ** 2
            RMSE = cp.sqrt(
                cp.sum(retrievedI - targetI) ** 2 / cp.sum(targetI) ** 2
            )
            RMSEList.append(float(RMSE))

            if iterTarget[0] == 0:
                # RMSE小于等于设置阈值
                if RMSEList[-1] <= iterTarget[1]:
                    break

        # 显存GC
        cp._default_memory_pool.free_all_blocks()

        return aK, phase
