import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage
from lib.cam.nncam import nncam


class CameraMiddleware(QObject):
    """
    [相机类] 相机控制中间件

    :var device: 相机对象
    :var imgWidth: 图像宽度
    :type imgWidth: int
    :var imgHeight: 图像高度
    :type imgHeight: int
    :var frame: 向UI线程推送的图像帧
    :var snapshot: 向UI线程推送的已抓取静态帧
    """

    frameUpdate = pyqtSignal()
    snapUpdate = pyqtSignal()
    expoUpdate = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.device = None
        self.imgWidth = 0
        self.imgHeight = 0
        self.frame = None
        self.snapshot = None
        self.targetFromSnap = False

        self._streamBuf = None
        self._snapBuf = None
        self.RESOLUTION = 2  # 1824x1216

    def openCamera(self):
        """
        [相机类] 打开相机硬件

        返回 0 为成功，-1 为失败
        """
        self.device = nncam.Nncam.Open(None)

        if self.device:
            # 分辨率
            self.device.put_eSize(self.RESOLUTION)

            self.imgWidth, self.imgHeight = self.device.get_Size()
            self.device.put_Option(nncam.NNCAM_OPTION_BYTEORDER, 0)  # Qimage use RGB byte order
            self.device.put_AutoExpoEnable(0)

            self._streamBuf = bytes(nncam.TDIBWIDTHBYTES(self.imgWidth * 24) * self.imgHeight)
            self._snapBuf = bytes(nncam.TDIBWIDTHBYTES(self.imgWidth * 24) * self.imgHeight)
            self.expoUpdateEvt()

            try:
                self.device.StartPullModeWithCallback(self.eventCallBack, self)
            except nncam.HRESULTException:
                self.closeCamera()
                return -1
            else:
                return 0
        else:
            return -1

    def closeCamera(self):
        """
        [相机类] 关闭相机硬件
        """
        if self.device:
            self.device.Close()

        self.device = None
        self._streamBuf = None

    @staticmethod
    def eventCallBack(nEvent, self):
        """
        [相机类] 处理来自nncam.dll或libnncam.so内置线程的触发事件
        """
        if self.device:
            # 动态帧事件，采集视频流（常规模式）
            if nEvent == nncam.NNCAM_EVENT_IMAGE:
                self.dynamicFramesEvt()
            # 静态帧事件，Snap()方法触发
            elif nEvent == nncam.NNCAM_EVENT_STILLIMAGE:
                self.stillFrameEvt()
            # 曝光调整事件，目前仅用于更新UI线程的曝光参数信息
            elif nEvent == nncam.NNCAM_EVENT_EXPOSURE:
                self.expoUpdateEvt()
            # 内部错误事件
            elif nEvent == nncam.NNCAM_EVENT_ERROR:
                self.closeCamera()

    def dynamicFramesEvt(self):
        """
        [相机类] 向UI线程推送动态帧
        """
        try:
            self.device.PullImageV3(self._streamBuf, 0, 24, 0, None)
        except nncam.HRESULTException:
            pass
        else:
            self.frame = QImage(
                self._streamBuf, self.imgWidth, self.imgHeight, QImage.Format.Format_RGB888
            )
            self.frameUpdate.emit()

    def stillFrameEvt(self):
        """
        [相机类] 向UI线程推送已抓取静态帧
        """
        info = nncam.NncamFrameInfoV3()
        try:
            self.device.PullImageV3(None, 1, 24, 0, info)  # peek
        except nncam.HRESULTException:
            pass
        else:
            if info.width > 0 and info.height > 0:
                try:
                    self.device.PullImageV3(self._snapBuf, 1, 24, 0, info)
                except nncam.HRESULTException:
                    pass
                else:
                    self.snapshot = np.frombuffer(self._snapBuf, np.uint8).reshape((info.height, info.width, 3))
                    self.snapUpdate.emit()

    def expoUpdateEvt(self):
        """
        [相机类] 向UI线程发送曝光值更新信号
        """
        self.expoUpdate.emit()
