import sys
import ctypes
import time
import serial
import cv2
import cupy as cp
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal, QSize, qInstallMessageHandler, QRect
from PyQt6.QtGui import QGuiApplication, QPixmap, QIcon, QImage, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, \
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, \
    QFileDialog, QMessageBox, \
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar
from lib.utils.utils import Utils, ImgProcess
from lib.holo.libHoloAlgmGPU import WCIA
from lib.holo.libHoloEssential import Holo, HoloCalcWorker
from lib.cam.camAPI import CameraMiddleware
from lib.laser.laserAPI import LaserMiddleWare


class VideoFrameWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.frame = QImage()
        self.marks = []
        self.currMark = None

    def paintEvent(self, event):
        # 获取绘图设备
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 绘制视频帧
        if not self.frame.isNull():
            windowAspect = self.width() / self.height()
            imageAspect = self.frame.width() / self.frame.height()

            # 根据宽高比确定缩放比例
            if windowAspect < imageAspect:
                scaleFactor = self.width() / self.frame.width()
            else:
                scaleFactor = self.height() / self.frame.height()

            # 计算缩放后的图像尺寸
            scaledSize = QSize(
                int(self.frame.width() * scaleFactor), int(self.frame.height() * scaleFactor)
            )

            # 缩放图像
            pixmap = QPixmap.fromImage(
                self.frame.scaled(
                    scaledSize,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            # 计算图像绘制的起始位置以居中显示
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2

            # 绘制缩放后的图像
            painter.drawPixmap(x, y, pixmap)

            # 绘制标记
            for mark in self.marks:
                painter.setPen(QPen(Qt.GlobalColor.white, 2))  # 设置标记颜色和线宽
                painter.drawRect(mark)

            painter.end()
        else:
            painter.end()

    def setPixmap(self, frame):
        # 更新视频帧
        self.frame = frame
        self.repaint()  # 重绘整个窗口

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 在鼠标点击位置添加标记
            self.marks.append(QRect(event.pos().x() - 10, event.pos().y() - 10, 20, 20))
            self.repaint()  # 重绘标记

        if event.button() == Qt.MouseButton.RightButton:
            for mark in self.marks:
                if mark.contains(event.pos()):
                    self.currMark = mark
                    break
            else:  # 如果没有标记被选中，重置选中状态
                self.currMark = None

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.currMark:
            self.marks.remove(self.currMark)
            self.currMark = None  # 重置选中状态
            self.update()  # 重绘窗口以反映变化


class MainWindow(QMainWindow):
    """
    [主窗口类] 包含预览与控制面板

    :var pendingImg: 预处理前的图像
    :var targetImg: 全息图目标图像
    :var holoU: 全息图复光场
    :var holoImg: 全息图(相位项转位图)
    :var holoImgRotated: 针对LCOS旋转方向的全息图
    """
    holoImgReady = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.pendingImg = None
        self.targetImg = None
        self.holoImg = None
        self.holoU = None
        self.holoImgRotated = None
        self._snapAsTarget = False
        self._uniList = []
        self._effiList = []
        self._RMSEList = []

        # 相机实例通信
        self.cam = CameraMiddleware()
        self.cam.frameUpdate.connect(lambda: self.camPreview.setPixmap(self.cam.frame))
        self.cam.snapUpdate.connect(self.snapRefreshEvent)
        self.cam.expoUpdate.connect(self.expTimeUpdatedEvent)

        # 副屏实例通信
        self.secondWin = SecondMonitorWindow()
        self.holoImgReady.connect(self.secondWin.displayHoloImg)

        self._initUI()

        if any(arg == '-ac' or arg == '--auto-open-camera' for arg in sys.argv[1:]):
            self.toggleCam()
            logHandler.info("Camera started automatically.")

    def _initUI(self):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SLM Holograph Generator")
        self.setWindowFlags(Qt.WindowType.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("../res/slm.ico"))
        self.setWindowTitle("SLM Holograph Generator")
        self.resize(1200, 600)

        # ==== 控制区域 ====

        # 输入功能区
        self.snapFromCamBtn = QPushButton('从相机捕获')
        self.snapFromCamBtn.clicked.connect(self.snapFromCam)
        self.snapFromCamBtn.setEnabled(False)

        openTargetFileBtn = QPushButton('新的目标图...')
        openTargetFileBtn.clicked.connect(self.openTargetImg)

        openHoloFileBtn = QPushButton('已有全息图...')
        openHoloFileBtn.clicked.connect(self.openHoloImg)

        inputTipsText = QLabel("欲载入已有全息图，相同目录下需存在保存全息图时生成的同名.npy文件")
        inputTipsText.setStyleSheet("color:#999")
        inputTipsText.setWordWrap(True)

        inputImgLayout = QGridLayout()
        inputImgLayout.addWidget(openTargetFileBtn, 0, 0, 1, 1)
        inputImgLayout.addWidget(openHoloFileBtn, 0, 1, 1, 1)
        inputImgLayout.addWidget(self.snapFromCamBtn, 1, 0, 1, 2)
        inputImgLayout.addWidget(inputTipsText, 2, 0, 1, 2)
        inputImgLayout.setColumnStretch(0, 1)
        inputImgLayout.setColumnStretch(1, 1)

        inputImgGroupBox = QGroupBox("输入")
        inputImgGroupBox.setLayout(inputImgLayout)

        # 全息图计算功能区
        holoAlgmText = QLabel("全息算法")

        self.holoAlgmSel = QComboBox()
        self.holoAlgmSel.addItem(f"WCIA")
        self.holoAlgmSel.setEnabled(False)

        initPhaseText = QLabel("初始相位")

        self.initPhaseSel = QComboBox()
        self.initPhaseSel.addItem(f"随机")
        self.initPhaseSel.addItem(f"目标光场IFFT")
        self.initPhaseSel.addItem(f"(WIP) 带限初始相位")
        self.initPhaseSel.setEnabled(False)

        maxIterNumText = QLabel("最大迭代")

        self.maxIterNumInput = QSpinBox()
        self.maxIterNumInput.setRange(0, 10000)
        self.maxIterNumInput.setValue(40)
        self.maxIterNumInput.setEnabled(False)

        iterTargetText = QLabel("终止迭代条件（%）")

        self.iterTargetSel = QComboBox()
        self.iterTargetSel.addItem(f"均方根误差 <=")

        self.iterTargetInput = QDoubleSpinBox()
        self.iterTargetInput.setRange(0, 100)
        self.iterTargetInput.setValue(1)
        self.iterTargetInput.setEnabled(False)

        enableAmpEncText = QLabel("编码振幅信息")
        self.enableAmpEncChk = QCheckBox()
        self.enableAmpEncChk.clicked.connect(
            lambda: self.orderInput.setEnabled(self.enableAmpEncChk.isChecked())
        )
        self.enableAmpEncChk.setEnabled(False)
        self.enableAmpEncChk.setChecked(False)

        orderInputText = QLabel("衍射阶数")
        self.orderInput = QDoubleSpinBox()
        self.orderInput.setRange(0, 10)
        self.orderInput.setValue(2)
        self.orderInput.setEnabled(False)

        self.calcHoloBtn = QPushButton(QIcon(QPixmap('../res/svg/calculator.svg')), ' 计算全息图')
        self.calcHoloBtn.clicked.connect(self.calcHoloImg)
        self.calcHoloBtn.setEnabled(False)

        self.saveHoloBtn = QPushButton(QIcon(QPixmap('../res/svg/download.svg')), ' 保存全息图...')
        self.saveHoloBtn.clicked.connect(self.saveHoloImg)
        self.saveHoloBtn.setEnabled(False)

        calHoloLayout = QGridLayout()
        calHoloLayout.addWidget(holoAlgmText, 0, 0, 1, 2)
        calHoloLayout.addWidget(self.holoAlgmSel, 0, 2, 1, 4)
        calHoloLayout.addWidget(initPhaseText, 1, 0, 1, 2)
        calHoloLayout.addWidget(self.initPhaseSel, 1, 2, 1, 4)
        calHoloLayout.addWidget(maxIterNumText, 2, 0, 1, 2)
        calHoloLayout.addWidget(self.maxIterNumInput, 2, 2, 1, 4)
        calHoloLayout.addWidget(iterTargetText, 3, 0, 1, 6)
        calHoloLayout.addWidget(self.iterTargetSel, 4, 0, 1, 4)
        calHoloLayout.addWidget(self.iterTargetInput, 4, 4, 1, 2)
        calHoloLayout.addWidget(enableAmpEncText, 5, 0, 1, 2)
        calHoloLayout.addWidget(self.enableAmpEncChk, 5, 2, 1, 1)
        calHoloLayout.addWidget(orderInputText, 5, 3, 1, 1)
        calHoloLayout.addWidget(self.orderInput, 5, 4, 1, 2)
        calHoloLayout.addWidget(self.calcHoloBtn, 6, 0, 1, 6)
        calHoloLayout.addWidget(self.saveHoloBtn, 7, 0, 1, 6)
        calHoloLayout.setColumnStretch(0, 1)
        calHoloLayout.setColumnStretch(1, 1)
        calHoloLayout.setColumnStretch(2, 1)
        calHoloLayout.setColumnStretch(3, 1)
        calHoloLayout.setColumnStretch(4, 1)
        calHoloLayout.setColumnStretch(5, 1)

        calHoloGroupBox = QGroupBox("全息图计算")
        calHoloGroupBox.setLayout(calHoloLayout)

        # 相机功能区
        self.toggleCamBtn = QPushButton('打开')
        self.toggleCamBtn.clicked.connect(self.toggleCam)

        self.snapBtn = QPushButton('抓图')
        self.snapBtn.clicked.connect(self.snapImg)
        self.snapBtn.setEnabled(False)

        expTimeText = QLabel("曝光 (ms)")

        self.expTimeInput = QSpinBox()
        self.expTimeInput.setEnabled(False)
        self.expTimeInput.textChanged.connect(self.expTimeSet)

        camCtrlLayout = QGridLayout()
        camCtrlLayout.addWidget(self.toggleCamBtn, 0, 0, 1, 2)
        camCtrlLayout.addWidget(self.snapBtn, 0, 2, 1, 4)
        camCtrlLayout.addWidget(expTimeText, 1, 0, 1, 2)
        camCtrlLayout.addWidget(self.expTimeInput, 1, 2, 1, 4)
        camCtrlLayout.setColumnStretch(0, 1)
        camCtrlLayout.setColumnStretch(1, 1)
        camCtrlLayout.setColumnStretch(2, 1)
        camCtrlLayout.setColumnStretch(3, 1)
        camCtrlLayout.setColumnStretch(4, 1)
        camCtrlLayout.setColumnStretch(5, 1)

        camCtrlGroupBox = QGroupBox("相机")
        camCtrlGroupBox.setLayout(camCtrlLayout)

        ctrlAreaLayout = QVBoxLayout()
        ctrlAreaLayout.addWidget(inputImgGroupBox)
        ctrlAreaLayout.addWidget(calHoloGroupBox)
        ctrlAreaLayout.addStretch(1)
        ctrlAreaLayout.addWidget(camCtrlGroupBox)
        if not any(arg == '-bl' or arg == '--bypass-laser-detection' for arg in sys.argv[1:]):
            self._initLaserUI()
            ctrlAreaLayout.addWidget(self.laserCtrlGroupBox)
        else:
            logHandler.warning("Bypass laser detection mode, laser control component will not be loaded.")

        ctrlArea = QWidget()
        ctrlArea.setObjectName("ctrlArea")
        ctrlArea.setLayout(ctrlAreaLayout)
        ctrlArea.setFixedWidth(280)

        # ==== 显示区域 ====
        # 相机预览区域
        self.camPreview = VideoFrameWidget()
        # self.camPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        # self.camPreview.setText("点击 [打开相机] 以预览...")
        self.camPreview.setMinimumSize(480, 360)

        camPreviewHLayout = QHBoxLayout()
        camPreviewHLayout.addWidget(self.camPreview)
        camPreviewVLayout = QVBoxLayout()
        camPreviewVLayout.addLayout(camPreviewHLayout)
        camPreviewGroupBox = QGroupBox("相机预览")
        camPreviewGroupBox.setLayout(camPreviewVLayout)

        # 目标图显示区域
        self.targetImgPreview = QLabel()
        self.targetImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.targetImgPreview.setText("从输入模块载入已有目标图 \n或打开相机后 [从相机捕获]")
        self.targetImgPreview.setMaximumSize(180, 180)

        targetImgHLayout = QHBoxLayout()
        targetImgHLayout.addWidget(self.targetImgPreview)
        targetImgVLayout = QVBoxLayout()
        targetImgVLayout.addLayout(targetImgHLayout)
        targetImgWidget = QGroupBox("目标图")
        targetImgWidget.setLayout(targetImgVLayout)

        # 全息图显示区域
        self.holoImgPreview = QLabel()
        self.holoImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.holoImgPreview.setText("从输入模块载入已有全息图 \n或从目标图 [计算全息图]")
        self.holoImgPreview.setMaximumSize(180, 180)

        holoImgHLayout = QHBoxLayout()
        holoImgHLayout.addWidget(self.holoImgPreview)
        holoImgVLayout = QVBoxLayout()
        holoImgVLayout.addLayout(holoImgHLayout)
        holoImgWidget = QGroupBox("全息图")
        holoImgWidget.setLayout(holoImgVLayout)

        # 重建光场仿真显示区域
        self.reconstructImgPreview = QLabel()
        self.reconstructImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.reconstructImgPreview.setText(f"计算或载入全息图后查看结果")
        self.reconstructImgPreview.setMaximumSize(180, 180)

        reconstructImgHLayout = QHBoxLayout()
        reconstructImgHLayout.addWidget(self.reconstructImgPreview)
        reconstructImgVLayout = QVBoxLayout()
        reconstructImgVLayout.addLayout(reconstructImgHLayout)
        reconstructImgWidget = QGroupBox("重建预览")
        reconstructImgWidget.setLayout(reconstructImgVLayout)

        calcInfoLayout = QVBoxLayout()
        calcInfoLayout.addWidget(targetImgWidget)
        calcInfoLayout.addWidget(holoImgWidget)
        calcInfoLayout.addWidget(reconstructImgWidget)

        calcInfoWidget = QWidget()
        calcInfoWidget.setLayout(calcInfoLayout)

        # ==== 状态栏 ====

        self.progressBar = QProgressBar()
        self.progressBar.setStyleSheet("QProgressBar {min-width: 100px; max-width: 200px; margin-right:5px}")
        self.progressBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.secondStatusInfo = QLabel()
        self.secondStatusInfo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.secondStatusInfo.setStyleSheet("QLabel {min-width: 100px; margin-right:5px}")

        self.statusBar = QStatusBar()
        self.statusBar.addPermanentWidget(self.secondStatusInfo)
        self.statusBar.addPermanentWidget(self.progressBar)

        self.statusBar.showMessage(f"就绪")
        self.secondStatusInfo.setText(f"等待输入")
        self.progressBar.setRange(0, 10)
        self.progressBar.setValue(0)

        # ==== 窗体 ====
        comboArea = QTabWidget()
        comboArea.addTab(ctrlArea, "控制")
        comboArea.addTab(calcInfoWidget, "视图")

        layout = QHBoxLayout()
        layout.addWidget(comboArea)
        layout.addWidget(camPreviewGroupBox)
        layout.setStretch(0, 1)
        layout.setStretch(1, 5)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setStatusBar(self.statusBar)

    def _initLaserUI(self):
        # 激光器功能区
        self.laserPortSel = QComboBox()
        self.laserPortSel.setEnabled(False)

        self.connectLaserBtn = QPushButton(' 连接')
        self.connectLaserIcon = QIcon(QPixmap('../res/svg/plug-fill.svg'))
        self.disconnectLaserIcon = QIcon(QPixmap('../res/svg/eject-fill.svg'))
        self.connectLaserBtn.setIcon(self.connectLaserIcon)
        self.connectLaserBtn.setIconSize(QSize(14, 14))
        self.connectLaserBtn.clicked.connect(self.toggleLaserConnection)
        self.connectLaserBtn.setEnabled(False)

        self.toggleLaserBtn = QPushButton(' 启动激光')
        self.emitLaserIcon = QIcon(QPixmap('../res/svg/sun-fill.svg'))
        self.stopLaserIcon = QIcon(QPixmap('../res/svg/x-octagon-fill.svg'))
        self.toggleLaserBtn.setIcon(self.emitLaserIcon)
        self.toggleLaserBtn.setIconSize(QSize(14, 14))
        self.toggleLaserBtn.clicked.connect(self.toggleLaserEmit)
        self.toggleLaserBtn.setEnabled(False)

        laserPwrText = QLabel("功率 (mW)")

        self.laserPwrInput = QSpinBox()
        self.laserPwrInput.setRange(0, 300)
        self.laserPwrInput.setValue(10)
        self.laserPwrInput.setEnabled(False)

        self.setLaserPwrBtn = QPushButton(' 设定')
        confirmIcon = QIcon(QPixmap('../res/svg/check-lg.svg'))
        self.setLaserPwrBtn.setIcon(confirmIcon)
        self.setLaserPwrBtn.setIconSize(QSize(14, 14))
        self.setLaserPwrBtn.clicked.connect(self.setLaserPwr)
        self.setLaserPwrBtn.setEnabled(False)

        laserCtrlLayout = QGridLayout()
        laserCtrlLayout.addWidget(self.laserPortSel, 0, 0, 1, 6)
        laserCtrlLayout.addWidget(self.connectLaserBtn, 1, 0, 1, 2)
        laserCtrlLayout.addWidget(self.toggleLaserBtn, 1, 2, 1, 4)
        laserCtrlLayout.addWidget(laserPwrText, 2, 0, 1, 2)
        laserCtrlLayout.addWidget(self.laserPwrInput, 2, 2, 1, 2)
        laserCtrlLayout.addWidget(self.setLaserPwrBtn, 2, 4, 1, 2)
        laserCtrlLayout.setColumnStretch(0, 1)
        laserCtrlLayout.setColumnStretch(1, 1)
        laserCtrlLayout.setColumnStretch(2, 1)
        laserCtrlLayout.setColumnStretch(3, 1)
        laserCtrlLayout.setColumnStretch(4, 1)
        laserCtrlLayout.setColumnStretch(5, 1)

        self.laserCtrlGroupBox = QGroupBox("激光器")
        self.laserCtrlGroupBox.setLayout(laserCtrlLayout)

        self.laser = LaserMiddleWare()
        self.deviceUpdatedEvent()

    def closeEvent(self, event):
        """
        [UI事件] 窗口关闭时同时关闭其他窗口，关闭相机
        """
        for widget in QApplication.instance().allWidgets():
            if isinstance(widget, QWidget) and widget != self:
                widget.close()

        self.cam.closeCamera()
        if not any(arg == '-bl' or arg == '--bypass-laser-detection' for arg in sys.argv[1:]):
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
            self.connectLaserBtn.setText(' 连接')
            self.connectLaserBtn.setIcon(self.connectLaserIcon)
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
                        f'未能打开激光器控制盒。\n '
                        f'详细信息：{err} \n'
                        f'请尝试重新连接USB线，并确认端口选择正确、驱动安装正确或未被其他程序占用。'
                    )
                    logHandler.error(f"Unable to open laser. {err}")
                else:
                    logHandler.info(f"Laser connected. Device at {self.laser.port}.")
                    self.statusBar.showMessage(f"激光器控制盒已连接 ({self.laser.port})")
                    self.connectLaserBtn.setText(' 断开')
                    self.connectLaserBtn.setIcon(self.disconnectLaserIcon)

        self.laserPortSel.setEnabled(not self.laserPortSel.isEnabled())
        self.toggleLaserBtn.setEnabled(not self.laserPortSel.isEnabled())
        if self.toggleLaserBtn.isEnabled():
            self.toggleLaserBtn.setStyleSheet("color: #000000; background-color: #f38d05;")
        else:
            self.toggleLaserBtn.setStyleSheet("")
        self.laserPwrInput.setEnabled(not self.laserPortSel.isEnabled())
        self.setLaserPwrBtn.setEnabled(not self.laserPortSel.isEnabled())

    def toggleLaserEmit(self):
        """
        [UI操作] 点击激光器出光
        """
        if self.laser.isEmitting:
            self.laser.setConfig('OFF')
            logHandler.info(f"Laser Config send. config: OFF")
            self.toggleLaserBtn.setText(' 启动激光')
            self.toggleLaserBtn.setIcon(self.emitLaserIcon)
            self.toggleLaserBtn.setStyleSheet("color: #000000; background-color: #f38d05;")
            self.statusBar.showMessage(f"激光器已停止")
        else:
            self.laser.setConfig('ON')
            logHandler.info(f"Laser Config send. config: ON")
            self.toggleLaserBtn.setText(' 停止激光')
            self.toggleLaserBtn.setIcon(self.stopLaserIcon)
            self.toggleLaserBtn.setStyleSheet("color: #ffffff; background-color: #ff0000;")
            self.statusBar.showMessage(f"激光器已出光，请注意操作安全。")

        self.connectLaserBtn.setEnabled(not self.connectLaserBtn.isEnabled())

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
            self.statusBar.showMessage(f"相机已停止")

            # self.camPreview.setPixmap(QImage(64, 64, QImage.Format.Format_RGB888).fill(Qt.GlobalColor.white))
            # self.camPreview.setText("点击 [打开相机] 以预览...")

            self.toggleCamBtn.setText("打开")
        else:
            result = self.cam.openCamera()
            if result == -1:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未能打开相机。\n'
                    f'尝试重新连接相机，并确认相机未被其他程序占用。'
                )
                logHandler.error(f"Unable to open camera. ")
                return -1
            else:
                logHandler.info(f"Camara opened. Resolution: {self.cam.imgWidth}x{self.cam.imgHeight}")
                self.statusBar.showMessage(f"相机分辨率 {self.cam.imgWidth}x{self.cam.imgHeight}")

                self.toggleCamBtn.setText("关闭")
                expMin, expMax, expCur = self.cam.device.get_ExpTimeRange()
                self.expTimeInput.setRange(10, 1000)
                self.expTimeInput.setValue(int(expCur / 1000))

                self.cam.expoUpdateEvt()

        self.snapBtn.setEnabled(not self.snapBtn.isEnabled())
        self.snapFromCamBtn.setEnabled(not self.snapFromCamBtn.isEnabled())
        self.expTimeInput.setEnabled(not self.expTimeInput.isEnabled())

    def snapImg(self):
        """
        [UI操作] 点击抓图
        """
        if self.cam.device:
            self.cam.device.Snap(self.cam.RESOLUTION)

    def snapFromCam(self):
        """
        [UI操作] 点击从相机捕获
        """
        if self.cam.device:
            self._snapAsTarget = True
            self.cam.device.Snap(self.cam.RESOLUTION)

    def expTimeSet(self, value):
        """
        [UI操作] 处理用户设置曝光时间状态
        """
        if self.cam.device:
            self.cam.device.put_ExpoAGain(100)
            self.cam.device.put_ExpoTime(int(value) * 1000)

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
                    self.pendingImg = ImgProcess.loadImg(imgDir)
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
                    self.holoImg = ImgProcess.loadImg(imgDir)
                    self.holoU = np.load(f"{Path(imgDir).parent / Path(imgDir).stem}.npy")
                except IOError:
                    QMessageBox.critical(
                        self, '错误', f'文件打开失败\n请确认文件可读，且目录中存在同名.npy文件'
                    )
                    logHandler.error(
                        f"Fail to load image: I/O Error. "
                        f"Please confirm that the file is readable, "
                        f"and a .npy file with the same name exists in the directory."
                    )
                else:
                    self.imgLoadedEvent(None, imgDir)
                    logHandler.info(f"Light field {Path(imgDir).parent / Path(imgDir).stem}.npy loaded.")

    def saveHoloImg(self):
        """
        [UI操作] 保存全息图
        """
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
                    np.save(f"{Path(imgDir).parent}/{Path(imgDir).stem}.npy", self.holoU)
                    self.statusBar.showMessage(f"保存成功。图片位于{imgDir}，并在目录中存储了同名.npy文件。"
                                               f"该文件包含光场信息，请与图像一同妥善保存，切勿更名。")
                    logHandler.info(f"Holo image saved at {imgDir}")

    def calcHoloImg(self):
        """
        [UI操作] 计算全息图
        """
        # 已载入图像
        if self.pendingImg is not None:
            self.targetImg = self.pendingImg

            logHandler.info(f"Start Calculation.")
            self.statusBar.showMessage(f"开始计算...")
            self.progressBar.reset()
            self.secondStatusInfo.setText(f"归一化与类型转换...")
            self.progressBar.setValue(1)

            self._uniList.clear()
            self._effiList.clear()
            self._RMSEList.clear()

            maxIterNum = self.maxIterNumInput.value()
            iterTarget = self.iterTargetInput.value() * 0.01

            targetNormalized = self.targetImg / 255

            # CuPy类型转换 (NumPy->CuPy)
            target = cp.asarray(targetNormalized)

            # 计时
            self.secondStatusInfo.setText(f"创建计算实例...")
            self.progressBar.setValue(2)
            self.tStart = time.time()
            try:
                if self.holoAlgmSel.currentIndex() == 0:
                    self.algorithm = WCIA(
                        target, maxIterNum,
                        initPhase=(self.initPhaseSel.currentIndex(), None),
                        iterTarget=(0, iterTarget),
                        uniList=self._uniList,
                        effiList=self._effiList,
                        RMSEList=self._RMSEList
                    )

                self.secondStatusInfo.setText(f"开始迭代...")
                self.progressBar.setRange(0, 0)

                self.thread = HoloCalcWorker(self.algorithm)
                self.thread.resultSig.connect(self.calcResultUpdateEvent)
                self.thread.start()
            except Exception as err:
                logHandler.error(f"Err in iteration: {err}")
                QMessageBox.critical(self, '错误', f'迭代过程中发生异常：\n{err}')
                self.statusBar.showMessage(f"迭代过程中发生异常: {err}")
            else:
                pass

        else:
            self.statusBar.showMessage(f"未载入目标图")
            logHandler.warning(f"No target image loaded. ")

        if self.holoImg is not None:
            self.saveHoloBtn.setEnabled(True)
        else:
            self.saveHoloBtn.setEnabled(False)

    def calcResultUpdateEvent(self, u, phase):
        self.secondStatusInfo.setText(f"类型转换...")
        self.progressBar.setRange(0, 10)
        self.progressBar.setValue(5)
        # CuPy类型转换 (CuPy->NumPy)
        self.holoImg = cp.asnumpy(Holo.genHologram(phase))
        self.holoU = cp.asnumpy(u)

        self.holoImgRotated = cv2.rotate(self.holoImg, cv2.ROTATE_90_CLOCKWISE)

        tEnd = time.time()

        del self.algorithm

        self.secondStatusInfo.setText(f"发送全息图...")
        self.progressBar.setValue(7)
        # 在预览窗口显示计算好的全息图
        ImgProcess.cvImg2QPixmap(self.holoImgPreview, self.holoImg)
        # 向副屏发送计算好的全息图
        self.holoImgReady.emit(self.holoImgRotated)
        logHandler.info(f"Image has been transferred to the second monitor.")

        self.secondStatusInfo.setText(f"计算重建效果...")
        self.progressBar.setValue(8)
        self.reconstructResult(self.holoU, 50, 532e-6)

        self.secondStatusInfo.setText(f"进行性能统计...")
        self.progressBar.setValue(9)
        # 性能估计
        iteration = len(self._uniList)
        duration = round(tEnd - self.tStart, 2)
        uniformity = self._uniList[-1]
        efficiency = self._effiList[-1]
        RMSE = self._RMSEList[-1]

        self.secondStatusInfo.setText(f"完成")
        self.progressBar.setValue(10)
        logHandler.info(f"Finish Calculation.")
        logHandler.info(
            f"Iteration={iteration}, Duration={duration}s, "
            f"uniformity={uniformity}, efficiency={efficiency}, RMSE={RMSE}"
        )
        self.statusBar.showMessage(
            f"计算完成。迭代{iteration}次，时长 {duration}s，"
            f"均匀度{round(uniformity, 4)}，光场利用效率{round(efficiency, 4)}, "
            f"RMSE={round(RMSE, 4)}"
        )

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

            self.pendingImg = ImgProcess().createGaussianBeamMask(self.pendingImg, 6)

            self.imgLoadedEvent(f"Snap", None)

            self._snapAsTarget = False

            filepath = f"../pics/snap/{time.strftime('%Y%m%d%H%M%S')}-asTarget.jpg"
        else:
            filepath = f"../pics/snap/{time.strftime('%Y%m%d%H%M%S')}.jpg"

        Utils.folderPathCheck(filepath)

        self.cam.snapshot.save(f"{filepath}")
        logHandler.info(f"Snapshot saved as '{filepath}'")
        self.statusBar.showMessage(f"截图已保存至 '{filepath}'")

    def imgLoadedEvent(self, targetDir, holoDir):
        """
        [UI事件] 载入图像后界面刷新
        """
        if targetDir is not None and targetDir != '':  # 载入目标图
            if self.pendingImg is not None:
                imgRes = f"{self.pendingImg.shape[1]}x{self.pendingImg.shape[0]}"
                logHandler.info(
                    f"Image Loaded. Dir={targetDir}, Resolution={imgRes}"
                )

                self.statusBar.showMessage(f"已加载图像{targetDir}，分辨率{imgRes}")
                self.holoImg = None
                ImgProcess.cvImg2QPixmap(self.targetImgPreview, self.pendingImg)
                ImgProcess.cvImg2QPixmap(self.holoImgPreview, None)
                ImgProcess.cvImg2QPixmap(self.reconstructImgPreview, None)

                self.holoImgPreview.setText("从输入模块载入已有全息图 \n或从目标图 [计算全息图]")
                self.reconstructImgPreview.setText(f"计算或载入全息图后查看结果")

                self.calcHoloBtn.setEnabled(True)
                self.holoAlgmSel.setEnabled(True)
                self.initPhaseSel.setEnabled(True)
                self.maxIterNumInput.setEnabled(True)
                self.iterTargetSel.setEnabled(True)
                self.iterTargetInput.setEnabled(True)
                self.enableAmpEncChk.setEnabled(True)
                self.secondStatusInfo.setText(f"就绪")
                self.progressBar.reset()
                self.progressBar.setValue(0)

            self.saveHoloBtn.setEnabled(False)
        elif holoDir is not None and holoDir != '':  # 载入全息图
            if self.holoImg is not None:  # 有全息图
                imgRes = f"{self.holoImg.shape[1]}x{self.holoImg.shape[0]}"
                logHandler.info(
                    f"Image Loaded. Dir='{holoDir}', Resolution={imgRes}"
                )

                self.statusBar.showMessage(f"已加载图像{holoDir}，分辨率{imgRes}")
                self.pendingImg = None
                ImgProcess.cvImg2QPixmap(self.targetImgPreview, None)
                ImgProcess.cvImg2QPixmap(self.holoImgPreview, self.holoImg)
                self.reconstructResult(self.holoU, 50, 532e-6)

                self.targetImgPreview.setText("从输入模块载入已有目标图 \n或打开相机后 [从相机捕获]")

                self.holoImgRotated = cv2.rotate(
                    cv2.flip(self.holoImg, 1),
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )
                self.holoImgReady.emit(self.holoImgRotated)

                self.calcHoloBtn.setEnabled(False)
                self.holoAlgmSel.setEnabled(False)
                self.initPhaseSel.setEnabled(False)
                self.maxIterNumInput.setEnabled(False)
                self.iterTargetSel.setEnabled(False)
                self.iterTargetInput.setEnabled(False)
                self.enableAmpEncChk.setEnabled(True)
        else:
            logHandler.warning(f"No image loaded. ")

    def reconstructResult(self, holoU, d: int, wavelength: float):
        """
        [UI事件] 重建光场和相位
        :param holoU: 全息光场
        :param d: 像面距离
        :param wavelength: 激光波长
        """

        # 重建光场
        reconstructU = Holo.reconstruct(holoU, d, wavelength)
        reconstructA = Holo.normalize(np.abs(reconstructU)) * 255

        ImgProcess.cvImg2QPixmap(self.reconstructImgPreview, reconstructA.astype("uint8"))

    def expTimeUpdatedEvent(self):
        """
        [UI事件] 处理相机更新曝光时间状态
        """
        expTime = self.cam.device.get_ExpoTime()
        # 当相机回调参数时阻止用户侧输入
        with QSignalBlocker(self.expTimeInput):
            self.expTimeInput.setValue(int(expTime / 1000))

    def deviceUpdatedEvent(self):
        """
        [UI事件] 串口设备侦测
        """
        try:
            self.laser.listComPorts()
        except IndexError:
            logHandler.warning(f"No available COM ports found.")
            QMessageBox.warning(
                self,
                '警告',
                f'未发现可用端口，无法连接至激光器控制盒。\n'
                f'请尝试重新连接USB线，并确认端口选择正确、驱动安装正确或未被其他程序占用。\n'
                f'若实验设备无需使用带通信功能的激光器控制盒，请忽略本消息。'
            )
            self.laserPortSel.addItem(f"未发现可用端口")
        else:
            logHandler.info(f"Detected COM port(s):")
            for i in range(len(self.laser.portList)):
                logHandler.info(f"{self.laser.portList[i]}")
                self.laserPortSel.addItem(f"{self.laser.portList[i]}")

            self.laserPortSel.setEnabled(True)
            self.connectLaserBtn.setEnabled(True)


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

        if not any(arg == '-bs' or arg == '--bypass-LCOS-detection' for arg in sys.argv[1:]):
            self.show()
        else:
            logHandler.warning("Bypass LCOS detection mode, second window component will not be loaded.")

    def displayHoloImg(self, image):
        """
        [UI事件] 显示全息图

        :param object image:
        """
        logHandler.info(f"Image has been received from the main window.")
        ImgProcess.cvImg2QPixmap(self.holoImgFullScn, image)

    def monitorDetection(self):
        """
        [UI事件] 检测显示器
        """
        screens = QGuiApplication.screens()

        if len(screens) > 1:
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
                    f'程序被配置为[自动检测LCOS设备]模式，但未检测到名为 "JDC EDK" 的LCOS硬件。\n'
                    f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                )
                logHandler.error(
                    f"Auto detect mode, No LCOS named 'JDC EDK' detected. "
                    f"Check your connection, power status or screen configuration."
                )
                sys.exit()

            selectedMon = screens[self._selMonIndex]
            self.setGeometry(selectedMon.geometry().x(), selectedMon.geometry().y(),
                             selectedMon.size().width(), selectedMon.size().height())
            logHandler.info(f"Monitor {self._selMonIndex} selected.")

        else:
            if any(arg == '-bs' or arg == '--bypass-LCOS-detection' for arg in sys.argv[1:]):
                self.setGeometry(screens[0].geometry().x(), screens[0].geometry().y(),
                                 screens[0].size().width(), screens[0].size().height())
                logHandler.warning(f"Bypass LCOS detection mode, current monitor selected.")

                QMessageBox.warning(
                    self,
                    '警告',
                    f'程序被配置为[绕过LCOS设备检测]模式。\n'
                    f'当前监视器被设置为LCOS监视器，屏幕可能闪烁。该功能仅供开发时使用。'
                )

            else:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未检测到多显示器。\n'
                    f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                )
                logHandler.error(
                    f"No Multi-monitors Detected. "
                    f"Check your connection, power status or screen configuration."
                )
                sys.exit()


if __name__ == '__main__':
    args = Utils.getCmdOpt()
    logHandler = Utils.getLog()

    app = QApplication(sys.argv)
    qInstallMessageHandler(Utils.exceptionHandler)
    window = MainWindow()
    window.showMaximized()

    sys.excepthook = Utils.exceptHook
    sys.exit(app.exec())
