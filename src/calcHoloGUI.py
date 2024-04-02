import sys
import ctypes
import time
import serial
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal, qInstallMessageHandler
from PyQt6.QtGui import QGuiApplication, QPixmap, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, \
    QGridLayout, QVBoxLayout, QHBoxLayout, \
    QDialog, QFileDialog, QMessageBox, \
    QGroupBox, QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox
import cv2
import cupy as cp
import numpy as np
from lib.utils.utils import Utils
from lib.holo import libGS_GPU as libGS
from lib.cam.camAPI import CameraMiddleware
from lib.laser.laserAPI import LaserMiddleWare


def exception_handler(type, value, traceback):
    # 捕获异步异常并显示错误消息
    QMessageBox.critical(None, 'Error', f"{type}\n{value}\n{traceback}")


class MainWindow(QMainWindow):
    """
    [主窗口类] 包含预览与控制面板

    :var pendingImg: 预处理前的图像
    :var targetImg: 全息图目标图像
    :var holoImg: 全息图
    :var holoImgRotated: 针对LCOS旋转方向的全息图
    """
    holoImgReady = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.pendingImg = None
        self.targetImg = None
        self.holoImg = None
        self.holoImgRotated = None
        self._isBinarized = False
        self._snapAsTarget = False
        self.binarizedImg = None

        # 相机实例通信
        self.cam = CameraMiddleware()
        self.cam.frameUpdate.connect(self.frameRefreshEvent)
        self.cam.snapUpdate.connect(self.snapRefreshEvent)
        self.cam.expoUpdate.connect(self.expTimeUpdatedEvent)

        # 激光器实例
        self.laser = LaserMiddleWare()

        # 副屏实例通信
        self.secondWin = SecondMonitorWindow()
        self.holoImgReady.connect(self.secondWin.displayHoloImg)

        self._initUI()

        self.deviceUpdatedEvent()

    def _initUI(self):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SLM Holograph Generator")
        self.setWindowIcon(QIcon("../res/slm.ico"))
        self.setWindowTitle("SLM Holograph Generator")
        self.resize(1200, 600)

        mainWindowStyleSheet = '''
            QWidget#ctrlArea {
                max-width: 240px;
                margin: 0;
            }
            QWidget#ctrlArea QPushButton {
                height: 18px; 
            }
            QWidget#ctrlArea QSpinBox, QWidget#ctrlArea QDoubleSpinBox {
                height: 22px; 
            }
        '''

        # ==== 控制区域 ====
        # self.binarizeImgBtn = QPushButton('二值化原图', self)
        # self.binarizeImgBtn.clicked.connect(self.onBinarizeImgBtnClicked)
        # self.binarizeImgBtn.setEnabled(False)

        # 相机功能区
        self.toggleCamBtn = QPushButton('打开相机')
        self.toggleCamBtn.clicked.connect(self.toggleCam)

        self.snapBtn = QPushButton('抓图')
        self.snapBtn.clicked.connect(self.snapImg)
        self.snapBtn.setEnabled(False)

        expTimeText = QLabel("曝光时间 (ms)")

        self.expTimeInput = QSpinBox()
        self.expTimeInput.setEnabled(False)
        self.expTimeInput.textChanged.connect(self.expTimeSet)

        self.autoExpChk = QCheckBox()
        self.autoExpChk.setEnabled(False)
        self.autoExpChk.stateChanged.connect(self.autoExpSet)
        autoExpText = QLabel("自动曝光")

        autoExpLayout = QHBoxLayout()
        autoExpLayout.addStretch()
        autoExpLayout.addWidget(autoExpText, 0, Qt.AlignmentFlag.AlignRight)
        autoExpLayout.addSpacing(10)
        autoExpLayout.addWidget(self.autoExpChk, 0, Qt.AlignmentFlag.AlignRight)

        expTimeSetLayout = QGridLayout()
        expTimeSetLayout.addWidget(expTimeText, 0, 0, 2, 1)
        expTimeSetLayout.addWidget(self.expTimeInput, 1, 0, 1, 1)
        expTimeSetLayout.addLayout(autoExpLayout, 1, 1, 1, 1)
        expTimeSetLayout.setVerticalSpacing(0)
        expTimeSetLayout.setColumnStretch(0, 1)
        expTimeSetLayout.setColumnStretch(1, 1)

        camCtrlLayout = QVBoxLayout()
        camCtrlLayout.addWidget(self.toggleCamBtn)
        camCtrlLayout.addWidget(self.snapBtn)
        camCtrlLayout.addWidget(expTimeText)
        camCtrlLayout.addLayout(expTimeSetLayout)

        camCtrlGroupBox = QGroupBox("相机设置")
        camCtrlGroupBox.setLayout(camCtrlLayout)

        # 激光器功能区
        self.laserPortSel = QComboBox()

        self.connectLaserBtn = QPushButton('连接激光器')
        self.connectLaserBtn.clicked.connect(self.toggleLaserConnection)

        laserPwrText = QLabel("激光功率 (mW)")

        self.laserPwrInput = QSpinBox()
        self.laserPwrInput.setRange(0, 300)
        self.laserPwrInput.setValue(10)
        self.laserPwrInput.setEnabled(False)

        self.setLaserPwrBtn = QPushButton('设定')
        self.setLaserPwrBtn.clicked.connect(self.setLaserPwr)
        self.setLaserPwrBtn.setEnabled(False)

        laserPwrSetLayout = QGridLayout()
        laserPwrSetLayout.addWidget(laserPwrText, 0, 0, 2, 1)
        laserPwrSetLayout.addWidget(self.laserPwrInput, 1, 0, 1, 1)
        laserPwrSetLayout.addWidget(self.setLaserPwrBtn, 1, 1, 1, 1)
        laserPwrSetLayout.setVerticalSpacing(0)
        laserPwrSetLayout.setColumnStretch(0, 1)
        laserPwrSetLayout.setColumnStretch(1, 1)

        self.toggleLaserBtn = QPushButton('启动激光')
        self.toggleLaserBtn.clicked.connect(self.toggleLaserEmit)
        self.toggleLaserBtn.setStyleSheet("height: 36px; font-size: 18px")
        self.toggleLaserBtn.setEnabled(False)

        laserCtrlLayout = QVBoxLayout()
        laserCtrlLayout.addWidget(self.laserPortSel)
        laserCtrlLayout.addWidget(self.connectLaserBtn)
        laserCtrlLayout.addWidget(laserPwrText)
        laserCtrlLayout.addLayout(laserPwrSetLayout)
        laserCtrlLayout.addWidget(self.toggleLaserBtn)

        laserCtrlGroupBox = QGroupBox("激光器设置")
        laserCtrlGroupBox.setLayout(laserCtrlLayout)

        # 输入功能区
        self.snapFromCamBtn = QPushButton('从相机捕获')
        self.snapFromCamBtn.clicked.connect(self.snapFromCam)
        self.snapFromCamBtn.setEnabled(False)

        openTargetFileBtn = QPushButton('载入已有目标图...')
        openTargetFileBtn.clicked.connect(self.openTargetImg)

        openHoloFileBtn = QPushButton('载入已有全息图...')
        openHoloFileBtn.clicked.connect(self.openHoloImg)

        inputImgLayout = QVBoxLayout()
        inputImgLayout.addWidget(self.snapFromCamBtn)
        inputImgLayout.addWidget(openTargetFileBtn)
        inputImgLayout.addWidget(openHoloFileBtn)

        inputImgGroupBox = QGroupBox("输入")
        inputImgGroupBox.setLayout(inputImgLayout)

        # 全息图计算功能区
        maxIterNumText = QLabel("最大迭代次数")

        self.maxIterNumInput = QSpinBox()
        self.maxIterNumInput.setRange(0, 10000)
        self.maxIterNumInput.setValue(100)
        self.maxIterNumInput.setEnabled(False)

        uniThresNumText = QLabel("均匀度阈值")

        self.uniThresNumInput = QDoubleSpinBox()
        self.uniThresNumInput.setRange(0, 1)
        self.uniThresNumInput.setValue(0.95)
        self.uniThresNumInput.setEnabled(False)

        holoSetLayout = QGridLayout()
        holoSetLayout.addWidget(maxIterNumText, 0, 0, 1, 1)
        holoSetLayout.addWidget(uniThresNumText, 0, 1, 1, 1)
        holoSetLayout.addWidget(self.maxIterNumInput, 1, 0, 1, 1)
        holoSetLayout.addWidget(self.uniThresNumInput, 1, 1, 1, 1)
        holoSetLayout.setColumnStretch(0, 1)
        holoSetLayout.setColumnStretch(1, 1)

        self.calcHoloBtn = QPushButton('计算全息图')
        self.calcHoloBtn.clicked.connect(self.calcHoloImg)
        self.calcHoloBtn.setEnabled(False)

        self.saveHoloBtn = QPushButton('保存全息图...')
        self.saveHoloBtn.clicked.connect(self.saveHoloImg)
        self.saveHoloBtn.setEnabled(False)

        calHoloLayout = QVBoxLayout()
        calHoloLayout.addLayout(holoSetLayout)
        calHoloLayout.addWidget(self.calcHoloBtn)
        calHoloLayout.addWidget(self.saveHoloBtn)

        calHoloGroupBox = QGroupBox("全息图计算")
        calHoloGroupBox.setLayout(calHoloLayout)

        ctrlAreaLayout = QVBoxLayout()
        ctrlAreaLayout.addWidget(camCtrlGroupBox)
        ctrlAreaLayout.addWidget(laserCtrlGroupBox)
        ctrlAreaLayout.addStretch(1)
        ctrlAreaLayout.addWidget(inputImgGroupBox)
        ctrlAreaLayout.addWidget(calHoloGroupBox)

        ctrlArea = QWidget()
        ctrlArea.setObjectName("ctrlArea")
        ctrlArea.setLayout(ctrlAreaLayout)

        # ==== 显示区域 ====
        self.camPreview = QLabel()
        self.camPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.camPreview.setText("点击 [打开相机] 以预览...")
        self.camPreview.setMinimumSize(480, 360)

        camPreviewLayout = QHBoxLayout()
        camPreviewLayout.addWidget(self.camPreview)

        camPreviewGroupBox = QGroupBox("相机预览")
        camPreviewGroupBox.setLayout(camPreviewLayout)

        # 目标图显示区域
        self.targetImgPreview = QLabel()
        self.targetImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.targetImgPreview.setText("从输入模块载入已有目标图/全息图 \n或打开相机后 [从相机捕获]")
        self.targetImgPreview.setMinimumSize(360, 240)

        targetImgPreviewLayout = QHBoxLayout()
        targetImgPreviewLayout.addWidget(self.targetImgPreview)

        targetImgPreviewGroupBox = QGroupBox("目标图")
        targetImgPreviewGroupBox.setLayout(targetImgPreviewLayout)

        # 全息图显示区域
        self.holoImgPreview = QLabel()
        self.holoImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.holoImgPreview.setMinimumSize(360, 240)

        holoImgPreviewLayout = QHBoxLayout()
        holoImgPreviewLayout.addWidget(self.holoImgPreview)

        holoImgPreviewGroupBox = QGroupBox("全息图")
        holoImgPreviewGroupBox.setLayout(holoImgPreviewLayout)

        imgViewAreaLayout = QGridLayout()
        imgViewAreaLayout.addWidget(camPreviewGroupBox, 0, 1, 2, 1)
        imgViewAreaLayout.addWidget(targetImgPreviewGroupBox, 0, 0, 1, 1)
        imgViewAreaLayout.addWidget(holoImgPreviewGroupBox, 1, 0, 1, 1)
        imgViewAreaLayout.setColumnStretch(0, 1)
        imgViewAreaLayout.setColumnStretch(1, 5)

        imgViewArea = QWidget()
        imgViewArea.setObjectName("imgViewArea")
        imgViewArea.setLayout(imgViewAreaLayout)

        # ==== 窗体 ====
        layout = QHBoxLayout()
        layout.addWidget(ctrlArea)
        layout.addWidget(imgViewArea)

        widget = QWidget()
        widget.setLayout(layout)
        widget.setStyleSheet(mainWindowStyleSheet)
        self.setCentralWidget(widget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"就绪")

    def closeEvent(self, event):
        """
        [UI事件] 窗口关闭时同时关闭其他窗口，关闭相机
        """
        for widget in QApplication.instance().allWidgets():
            if isinstance(widget, QWidget) and widget != self:
                widget.close()

        self.cam.closeCamera()
        self.laser.closeComPort()
        logHandler.info(f"Bye.")
        event.accept()

    def toggleLaserConnection(self):
        """
        [UI操作] 点击连接激光器
        """
        if self.laser.device:
            self.laser.closeComPort()
            self.laser.device = None
            logHandler.info(f"Laser disconnected.")
            self.statusBar.showMessage(f"激光器已断开")
            self.connectLaserBtn.setText('连接激光器')
        else:
            selLaserPortIndex = self.laserPortSel.currentIndex()
            if self.laser.portList is not None and selLaserPortIndex is not None:
                self.laser.port = str(self.laser.portList[selLaserPortIndex].name)
                try:
                    self.laser.openComPort()
                except serial.serialutil.SerialException as err:
                    QMessageBox.critical(
                        self,
                        '错误',
                        f'未能打开激光器。\n '
                        f'详细信息：{err} \n'
                        f'请尝试重新连接USB线，并确认端口选择正确、驱动安装正确或未被其他程序占用。'
                    )
                    logHandler.error(f"Unable to open laser. {err}")
                else:
                    logHandler.info(f"Laser connected. Device at {self.laser.port}.")
                    self.statusBar.showMessage(f"激光器已连接 ({self.laser.port})")
                    self.connectLaserBtn.setText('断开激光器')

        self.laserPortSel.setEnabled(not self.laserPortSel.isEnabled())
        self.toggleLaserBtn.setEnabled(not self.laserPortSel.isEnabled())
        self.laserPwrInput.setEnabled(not self.laserPortSel.isEnabled())
        self.setLaserPwrBtn.setEnabled(not self.laserPortSel.isEnabled())
        logHandler.debug(f"UI thread updated. Initiator=User  Mode=toggle")

    def toggleLaserEmit(self):
        """
        [UI操作] 点击激光器出光
        """
        if self.laser.isEmitting:
            self.laser.setConfig('OFF')
            logHandler.info(f"Laser Config send. config: OFF")
            self.toggleLaserBtn.setText('启动激光')
            self.statusBar.showMessage(f"激光器已停止")
        else:
            self.laser.setConfig('ON')
            logHandler.info(f"Laser Config send. config: ON")
            self.toggleLaserBtn.setText('停止激光')
            self.statusBar.showMessage(f"激光器已出光，请注意操作安全。")

        self.connectLaserBtn.setEnabled(not self.connectLaserBtn.isEnabled())
        logHandler.debug(f"UI thread updated. Initiator=User  Mode=toggle")

    def setLaserPwr(self):
        """
        [UI操作] 点击设置激光器功率
        """
        if self.laser.device:
            self.laser.setConfig(self.laserPwrInput.value())
            self.statusBar.showMessage(f"激光器功率设定为 {self.laserPwrInput.value()}mW")
            logHandler.info(f"Laser Config send. config: pwr {self.laserPwrInput.value()}")

    def toggleCam(self):
        """
        [UI操作] 点击打开相机
        """
        if self.cam.device:
            self.cam.closeCamera()

            logHandler.info(f"Camara closed.")
            self.statusBar.showMessage(f"就绪")

            self.camPreview.clear()
            self.camPreview.setText("点击 [打开相机] 以预览...")

            self.toggleCamBtn.setText("打开相机")
        else:
            result = self.cam.openCamera()
            if result == -1:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未能打开相机。\n'
                    f'请尝试重新连接相机，并确认相机未被其他程序占用。'
                )
                logHandler.error(
                    f"Unable to open camera. "
                    f"Check your connection and affirm the camera is not occupied by other programs."
                )
                return -1
            else:
                logHandler.info(f"Camara opened. Resolution: {self.cam.imgWidth}x{self.cam.imgHeight}")
                self.statusBar.showMessage(f"相机分辨率 {self.cam.imgWidth}x{self.cam.imgHeight}")

                self.toggleCamBtn.setText("关闭相机")
                self.autoExpChk.setChecked(self.cam.device.get_AutoExpoEnable() == 1)
                expMin, expMax, expCur = self.cam.device.get_ExpTimeRange()
                self.expTimeInput.setRange(10, 1000)
                self.expTimeInput.setValue(int(expCur / 1000))

                self.cam.expoUpdateEvt()
                logHandler.debug(f"Exposure settings updated.")

        self.snapBtn.setEnabled(not self.snapBtn.isEnabled())
        self.snapFromCamBtn.setEnabled(not self.snapFromCamBtn.isEnabled())
        self.autoExpChk.setEnabled(not self.autoExpChk.isEnabled())
        self.expTimeInput.setEnabled(not self.expTimeInput.isEnabled())

        logHandler.debug(f"UI thread updated. Initiator=User  Mode=toggle")

    def snapImg(self):
        """
        [UI操作] 点击抓图
        """
        if self.cam.device:
            self.cam.device.Snap(self.cam.RESOLUTION)

            logHandler.debug(f"Snap Signal send. Initiator=User Mode=save")

    def snapFromCam(self):
        """
        [UI操作] 点击从相机捕获
        """
        if self.cam.device:
            self._snapAsTarget = True
            self.cam.device.Snap(self.cam.RESOLUTION)

            logHandler.debug(f"Snap Signal send. Initiator=User Mode=target)")

    def openTargetImg(self):
        """
        [UI操作] 点击载入已有图像
        """
        try:
            imgDir, imgType = QFileDialog.getOpenFileName(
                self, "打开图片", "", "图像文件(*.jpg *.png *.tif *.bmp);;所有文件(*)"
            )
        except Exception as err:
            QMessageBox.critical(self, '错误', f'载入图像失败：\n{err}')
            logHandler.error(f"Fail to load image: {err}")
        else:
            if imgDir is not None and imgDir != '':
                try:
                    self.pendingImg = Utils().loadImg(imgDir)
                except IOError:
                    QMessageBox.critical(self, '错误', '文件打开失败')
                    logHandler.error(f"Fail to load image: I/O Error")

            self.imgLoadedEvent(imgDir, None)

    def openHoloImg(self):
        """
        [UI操作] 点击载入已有图像
        """
        try:
            imgDir, imgType = QFileDialog.getOpenFileName(
                self, "打开图片", "", "图像文件(*.jpg *.png *.tif *.bmp);;所有文件(*)"
            )
        except Exception as err:
            QMessageBox.critical(self, '错误', f'载入图像失败：\n{err}')
            logHandler.error(f"Fail to load image: {err}")
        else:
            if imgDir is not None and imgDir != '':
                try:
                    self.holoImg = Utils().loadImg(imgDir)
                except IOError:
                    QMessageBox.critical(self, '错误', '文件打开失败')
                    logHandler.error(f"Fail to load image: I/O Error")

            self.imgLoadedEvent(None, imgDir)

    def saveHoloImg(self):
        if self.holoImg is not None:
            try:
                imgDir, imgType = QFileDialog.getSaveFileName(
                    self, "保存图片", "", "*.jpg;;*.png;;*.tif;;*.bmp"
                )
            except Exception as err:
                QMessageBox.critical(self, '错误', f'保存图像失败：\n{err}')
                logHandler.error(f"Fail to save image: {err}")
            else:
                if imgDir is not None and imgDir != '':
                    cv2.imencode(imgType, self.holoImg)[1].tofile(imgDir)
                    self.statusBar.showMessage(f"保存成功。图片位于{imgDir}")
                    logHandler.info(f"Holo image saved at {imgDir}")

    def autoExpSet(self, state):
        """
        [UI操作] 处理用户设置自动曝光状态
        """
        if self.cam.device:
            self.cam.device.put_AutoExpoEnable(1 if state else 0)
            logHandler.debug(f"Exp config send. Initiator=User")
            self.expTimeInput.setEnabled(not state)
            logHandler.debug(f"UI thread updated. Initiator=User  Mode=refresh")

    def expTimeSet(self, value):
        """
        [UI操作] 处理用户设置曝光时间状态
        """
        if self.cam.device:
            self.cam.device.put_ExpoAGain(100)
            if not self.autoExpChk.isChecked():
                self.cam.device.put_ExpoTime(int(value) * 1000)
                logHandler.debug(f"Exp config send. Initiator=User")

    def onBinarizeImgBtnClicked(self):
        """
        (WIP)[UI操作] 对待处理图像二值化

        todo: 解决二值化效果不佳
        """
        if self.pendingImg is not None:
            # 还原原图
            if self._isBinarized:
                Utils().cvImg2QPixmap(self.targetImgPreview, self.pendingImg)
                self.binarizeImgBtn.setText("二值化原图")
                self.statusBar.showMessage(f"已还原原图")
                self._isBinarized = False
                logHandler.info(f"Image has been restored.")
            # 二值化原图
            else:
                try:
                    self.binarizedImg = cv2.threshold(
                        self.pendingImg, 127, 255, cv2.THRESH_BINARY
                    )
                    # self.binarizedImg = cv2.adaptiveThreshold(
                    #     self.pendingImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    # )
                except Exception as err:
                    logHandler.error(f"Fail to binarize image: {err}")
                else:
                    Utils().cvImg2QPixmap(self.targetImgPreview, self.binarizedImg)
                    self.binarizeImgBtn.setText("还原原图")
                    self.statusBar.showMessage(f"已二值化图像")
                    self._isBinarized = True
                    logHandler.info(f"Image has been binarized.")

    def calcHoloImg(self):
        """
        [UI操作] 计算全息图
        """
        # 已载入图像
        if self.pendingImg is not None:
            if self._isBinarized:
                self.targetImg = self.binarizedImg
            else:
                self.targetImg = self.pendingImg

            logHandler.info(f"Start Calculation.")
            self.statusBar.showMessage(f"开始计算...")

            uniList = []
            targetNormalized = self.targetImg / 255

            maxIterNum = self.maxIterNumInput.value()
            uniThres = self.uniThresNumInput.value()

            # CuPy类型转换 (NumPy->CuPy)
            target = cp.asarray(targetNormalized)

            # 计时
            tStart = time.time()
            try:
                phase, normIntensity = libGS.GSiteration(maxIterNum, uniThres, target, uniList)
            except Exception as err:
                logHandler.error(f"Err in GSiteration: {err}")
                QMessageBox.critical(self, '错误', f'GS迭代过程中发生异常：\n{err}')
                self.statusBar.showMessage(f"GS迭代过程中发生异常: {err}")
            else:
                holo = libGS.genHologram(phase)

                # CuPy类型转换 (CuPy->NumPy)
                self.holoImg = cp.asnumpy(holo)

                self.holoImgRotated = cv2.rotate(
                    cv2.flip(self.holoImg, 1),
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )

                tEnd = time.time()

                # 显存GC
                cp._default_memory_pool.free_all_blocks()

                # 在预览窗口显示计算好的全息图
                Utils().cvImg2QPixmap(self.holoImgPreview, self.holoImg)
                # 向副屏发送计算好的全息图
                self.holoImgReady.emit(self.holoImgRotated)
                logHandler.info(f"Image has been transferred to the second monitor.")

                # 性能估计
                iteration = len(uniList)
                duration = round(tEnd - tStart, 2)
                uniformity = uniList[-1]
                efficiency = cp.sum(normIntensity[target == 1]) / cp.sum(target[target == 1])

                logHandler.info(f"Finish Calculation.")
                logHandler.info(
                    f"Iteration={iteration}, Duration={duration}s, "
                    f"uniformity={uniformity}, efficiency={efficiency}"
                )
                self.statusBar.showMessage(
                    f"计算完成。迭代{iteration}次，时长 {duration}s，"
                    f"均匀度{round(uniformity, 4)}，光场利用效率{cp.around(efficiency, 4)}"
                )
        else:
            self.statusBar.showMessage(f"未载入目标图")
            logHandler.warning(f"No target image loaded. ")

        if self.holoImg is not None:
            self.saveHoloBtn.setEnabled(True)
        else:
            self.saveHoloBtn.setEnabled(False)

    def frameRefreshEvent(self):
        """
        [UI事件] 刷新相机预览窗口
        """
        preview = self.cam.frame.scaled(
            self.camPreview.width(), self.camPreview.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        )
        self.camPreview.setPixmap(QPixmap.fromImage(preview))

    def snapRefreshEvent(self):
        """
        [UI事件] 抓图
        """
        if self._snapAsTarget:
            ptr = self.cam.snapshot.bits()
            ptr.setsize(self.cam.snapshot.bytesPerLine() * self.cam.snapshot.height())
            height = self.cam.snapshot.height()
            width = self.cam.snapshot.width()

            # 创建NumPy数组
            image_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))

            self.pendingImg = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            cv2.imwrite('./snap.jpg', self.pendingImg)
            self.imgLoadedEvent(f"Snap", None)
            self._snapAsTarget = False

            filename = f"{time.strftime('%Y%m%d%H%M%S')}-asTarget"
        else:
            filename = f"{time.strftime('%Y%m%d%H%M%S')}"

        self.cam.snapshot.save(f"../pics/snap/{filename}.jpg")
        logHandler.info(f"Snapshot saved as '../pics/snap/{filename}.jpg'")
        self.statusBar.showMessage(f"截图已保存至 '../pics/snap/{filename}.jpg'")

    def imgLoadedEvent(self, targetDir, holoDir):
        """
        [UI事件] 载入目标图像后界面刷新
        """
        if targetDir is not None and targetDir != '':
            if self.pendingImg is not None:
                imgRes = f"{self.pendingImg.shape[1]}x{self.pendingImg.shape[0]}"
                logHandler.info(
                    f"Image Loaded. Dir={targetDir}, Resolution={imgRes}"
                )

                self.statusBar.showMessage(f"已加载图像{targetDir}，分辨率{imgRes}")
                Utils().cvImg2QPixmap(self.targetImgPreview, self.pendingImg)
                self.holoImg = None
                Utils().cvImg2QPixmap(self.holoImgPreview, None)

                self._isBinarized = False
                # self.binarizeImgBtn.setText("二值化原图")

                self.calcHoloBtn.setEnabled(True)
                # self.binarizeImgBtn.setEnabled(True)
                self.maxIterNumInput.setEnabled(True)
                self.uniThresNumInput.setEnabled(True)
            else:
                self.calcHoloBtn.setEnabled(False)
                # self.binarizeImgBtn.setEnabled(False)
                self.maxIterNumInput.setEnabled(False)
                self.uniThresNumInput.setEnabled(False)

            self.saveHoloBtn.setEnabled(False)
            logHandler.debug(f"UI thread updated. Initiator=User  Mode=refresh")
        elif holoDir is not None and holoDir != '':
            if self.holoImg is not None:
                imgRes = f"{self.holoImg.shape[1]}x{self.holoImg.shape[0]}"
                logHandler.info(
                    f"Image Loaded. Dir={holoDir}, Resolution={imgRes}"
                )

                self.statusBar.showMessage(f"已加载图像{holoDir}，分辨率{imgRes}")
                Utils().cvImg2QPixmap(self.targetImgPreview, None)
                self.pendingImg = None
                Utils().cvImg2QPixmap(self.holoImgPreview, self.holoImg)

                self.holoImgRotated = cv2.rotate(self.holoImg, cv2.ROTATE_90_CLOCKWISE)
                self.holoImgReady.emit(self.holoImgRotated)

                self._isBinarized = False
                # self.binarizeImgBtn.setText("二值化原图")

                self.calcHoloBtn.setEnabled(False)
                # self.binarizeImgBtn.setEnabled(False)
                self.maxIterNumInput.setEnabled(False)
                self.uniThresNumInput.setEnabled(False)
            else:
                self.calcHoloBtn.setEnabled(False)
                # self.binarizeImgBtn.setEnabled(False)
                self.maxIterNumInput.setEnabled(False)
                self.uniThresNumInput.setEnabled(False)
        else:
            logHandler.warning(f"No image loaded. ")

    def expTimeUpdatedEvent(self):
        """
        [UI事件] 处理相机更新曝光时间状态
        """
        expTime = self.cam.device.get_ExpoTime()
        # 当相机回调参数时阻止用户侧输入
        with QSignalBlocker(self.expTimeInput):
            self.expTimeInput.setValue(int(expTime / 1000))
            logHandler.debug(f"UI thread updated. Initiator=Camera  Mode=refresh")

    def deviceUpdatedEvent(self):
        try:
            self.laser.listComPorts()
        except IndexError:
            logHandler.error(f"No available ports found.")
            QMessageBox.critical(
                self,
                '错误',
                f'未发现可用端口\n'
                f'请尝试重新连接USB线，并确认端口选择正确、驱动安装正确或未被其他程序占用。'
            )
        else:
            logHandler.info(f"Detected COM port(s):")
            for i in range(len(self.laser.portList)):
                logHandler.info(f"{self.laser.portList[i]}")
                self.laserPortSel.addItem(f"{self.laser.portList[i]}")


class SecondMonitorWindow(QMainWindow):
    """
    [副屏窗口类] 用于显示全息图
    """

    def __init__(self):
        super().__init__()

        self._selMonIndex = None
        self._initUI()

    def _initUI(self):
        self.setWindowIcon(QIcon("../res/slm.ico"))
        # 隐藏任务栏按钮
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)

        self.holoImgFullScn = QLabel(self)
        self.holoImgFullScn.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.holoImgFullScn.setMinimumSize(320, 320)
        self.holoImgFullScn.setStyleSheet("font-size: 14px; color: #fff")
        self.holoImgFullScn.setText("请在主窗口计算全息图")

        layout = QVBoxLayout()
        layout.addWidget(self.holoImgFullScn)
        layout.setContentsMargins(0, 0, 0, 0)

        widget = QWidget()
        widget.setLayout(layout)
        widget.setStyleSheet("QWidget{background-color: #000}")
        self.setCentralWidget(widget)

        self.monitorDetection()

        self.show()

    def displayHoloImg(self, image):
        """
        [UI事件] 显示全息图

        :param object image:
        """
        logHandler.info(f"Image has been received from the main window.")
        Utils().cvImg2QPixmap(self.holoImgFullScn, image)

    def monitorDetection(self):
        """
        [UI事件] 检测显示器
        """
        screens = QGuiApplication.screens()

        if len(screens) > 1:
            autodetect = any(arg == '-d' or arg == '--autodetect' for arg in sys.argv[1:])
            if autodetect:
                for index, screen in enumerate(screens):
                    if 'JDC EDK' in screen.name():
                        scrWid = screen.size().width()
                        scrHei = screen.size().height()
                        scrScale = screen.devicePixelRatio()
                        scrRes = f"{round(scrWid * scrScale)}x{round(scrHei * scrScale)}"
                        scrRefreshRate = round(screen.refreshRate())
                        logHandler.info(
                            f"JDC EDK Detected: Monitor {index} - {scrRes}@{scrRefreshRate}Hz"
                        )
                        self._selMonIndex = index
                        break
                else:
                    QMessageBox.critical(
                        self,
                        '错误',
                        f'未检测到名为 "JDC EDK" 的LCOS硬件。\n'
                        f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                    )
                    logHandler.error(
                        f"No LCOS named 'JDC EDK' Detected. "
                        f"Check your connection, power status or screen configuration."
                    )
                    sys.exit()
            else:
                self.monitorDetectionUI(screens)

            selectedMon = screens[self._selMonIndex]
            self.setGeometry(selectedMon.geometry().x(), selectedMon.geometry().y(),
                             selectedMon.size().width(), selectedMon.size().height())
            logHandler.info(f"Monitor {self._selMonIndex} selected.")

        else:
            QMessageBox.critical(
                self,
                '错误',
                f'未检测到多显示器。\n'
                f'请重新连接SLM，，确认电源已打开，或在显示设置中确认配置正确。'
            )
            logHandler.error(
                f"No Multi-monitors Detected. "
                f"Check your connection, power status or screen configuration."
            )
            sys.exit()

    def monitorDetectionUI(self, screens):
        """
        [UI操作] 手动选择显示器
        """
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择 SLM 所在显示器")
        dialog.setModal(True)

        scrSel = QComboBox()

        logHandler.info(f"Detected Monitor(s):")
        # 枚举显示器
        for index, screen in enumerate(screens):
            scrWid = screen.size().width()
            scrHei = screen.size().height()
            scrScale = screen.devicePixelRatio()
            scrRes = f"{round(scrWid * scrScale)}x{round(scrHei * scrScale)}"
            scrRefreshRate = round(screen.refreshRate())

            logHandler.info(f"{index} - {screen.name()} {scrRes}@{scrRefreshRate}Hz")
            scrSel.addItem(f"{screen.name()} - {scrRes}@{scrRefreshRate}Hz")

        selBtn = QPushButton('确定')
        selBtn.clicked.connect(dialog.accept)
        selBtn.setStyleSheet("height: 24px")

        dialogLayout = QVBoxLayout()
        dialogLayout.addWidget(scrSel)
        dialogLayout.addWidget(selBtn)

        dialog.setLayout(dialogLayout)

        result = dialog.exec()
        # 确认
        if result == 1:
            self._selMonIndex = scrSel.currentIndex()
        # 点x
        else:
            sys.exit()


if __name__ == '__main__':
    args = Utils().getCmdOpt()

    logHandler = Utils().getLog(
        consoleLevel=args.cloglvl,
        fileLevel=args.floglvl,
        writeLogFile=(args.floglvl > 0)
    )

    app = QApplication(sys.argv)
    qInstallMessageHandler(exception_handler)
    window = MainWindow()
    window.showMaximized()

    sys.excepthook = Utils.exceptHook
    sys.exit(app.exec())
