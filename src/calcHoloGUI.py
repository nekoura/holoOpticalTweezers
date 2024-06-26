import sys
import ctypes
import time
import serial
import cv2
import cupy as cp
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal, QSize, qInstallMessageHandler
from PyQt6.QtGui import QGuiApplication, QPixmap, QIcon
from PyQt6.QtWidgets import QApplication, QStyleFactory, QMainWindow, QWidget, QStatusBar, \
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, \
    QDialog, QFileDialog, QMessageBox, \
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar
from matplotlib import pyplot, ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from lib.utils.utils import Utils, ImgProcess
from lib.holo.libHoloAlgmGPU import GSW, WCIA
from lib.holo.libHoloEssential import Holo
from lib.cam.camAPI import CameraMiddleware
from lib.laser.laserAPI import LaserMiddleWare

pyplot.ion()


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
        self.binarizedImg = None
        self._uniList = []
        self._effiList = []
        self._RMSEList = []
        self._SSIMList = []

        self._pressed = False
        self._lastX = 0
        self._lastY = 0

        # 相机实例通信
        self.cam = CameraMiddleware()
        self.cam.frameUpdate.connect(self.frameRefreshEvent)
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
        self.setWindowIcon(QIcon("../res/slm.ico"))
        self.setWindowTitle("SLM Holograph Generator")
        self.resize(1200, 600)

        # ==== 控制区域 ====

        # 输入功能区
        self.snapFromCamBtn = QPushButton(' 从相机捕获')
        snapFromCamBtnIcon = QIcon(QPixmap('../res/svg/camera-fill.svg'))
        self.snapFromCamBtn.setIcon(snapFromCamBtnIcon)
        self.snapFromCamBtn.setIconSize(QSize(14, 14))
        self.snapFromCamBtn.clicked.connect(self.snapFromCam)
        self.snapFromCamBtn.setEnabled(False)

        openTargetFileBtn = QPushButton(' 新的目标图...')
        openTargetFileBtnIcon = QIcon(QPixmap('../res/svg/file-earmark-image.svg'))
        openTargetFileBtn.setIcon(openTargetFileBtnIcon)
        openTargetFileBtn.setIconSize(QSize(14, 14))
        openTargetFileBtn.clicked.connect(self.openTargetImg)

        openHoloFileBtn = QPushButton(' 已有全息图...')
        openHoloFileBtnIcon = QIcon(QPixmap('../res/svg/qr-code.svg'))
        openHoloFileBtn.setIcon(openHoloFileBtnIcon)
        openHoloFileBtn.setIconSize(QSize(14, 14))
        openHoloFileBtn.setStyleSheet("color: #ffffff; background-color: #2878B7;")
        openHoloFileBtn.clicked.connect(self.openHoloImg)

        inputTipsText = QLabel("欲载入已有全息图，相同目录下需存在保存全息图时生成的同名.npy文件")
        inputTipsText.setStyleSheet("color:#999")
        inputTipsText.setWordWrap(True)

        inputImgLayout = QGridLayout()
        inputImgLayout.addWidget(self.snapFromCamBtn, 0, 0, 1, 1)
        inputImgLayout.addWidget(openTargetFileBtn, 0, 1, 1, 1)
        inputImgLayout.addWidget(openHoloFileBtn, 1, 0, 1, 2)
        inputImgLayout.addWidget(inputTipsText, 2, 0, 1, 2)
        inputImgLayout.setColumnStretch(0, 1)
        inputImgLayout.setColumnStretch(1, 1)

        inputImgGroupBox = QGroupBox("输入")
        inputImgGroupBox.setLayout(inputImgLayout)

        # 全息图计算功能区
        holoAlgmText = QLabel("全息算法")

        self.holoAlgmSel = QComboBox()
        self.holoAlgmSel.addItem(f"WCIA")
        self.holoAlgmSel.addItem(f"(Deprecated) 加权GS")
        self.holoAlgmSel.addItem(f"(WIP) MRAF")
        self.holoAlgmSel.addItem(f"(WIP) PhaseOnly")
        self.holoAlgmSel.addItem(f"(WIP) RCWA")
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
        self.iterTargetSel.addItem(f"(实验性) SSIM >=")
        self.iterTargetSel.addItem(f"光能利用率 >=")
        self.iterTargetSel.addItem(f"光场均匀度 >=")
        self.iterTargetSel.setEnabled(False)

        self.iterTargetInput = QDoubleSpinBox()
        self.iterTargetInput.setRange(0, 100)
        self.iterTargetInput.setValue(1)
        self.iterTargetInput.setEnabled(False)

        self.enableSSIMChk = QCheckBox()
        self.enableSSIMChk.setEnabled(False)
        self.enableSSIMChk.setChecked(True)
        enableSSIMText = QLabel("计算SSIM")

        self.enableAmpEncChk = QCheckBox()
        self.enableAmpEncChk.setEnabled(False)
        self.enableAmpEncChk.setChecked(False)
        enableAmpEncText = QLabel("编码振幅信息")

        self.calcHoloBtn = QPushButton(' 计算全息图')
        calcHoloBtnIcon = QIcon(QPixmap('../res/svg/calculator.svg'))
        self.calcHoloBtn.setIcon(calcHoloBtnIcon)
        self.calcHoloBtn.setIconSize(QSize(14, 14))
        self.calcHoloBtn.clicked.connect(self.calcHoloImg)
        self.calcHoloBtn.setEnabled(False)

        self.saveHoloBtn = QPushButton(' 保存全息图...')
        saveHoloBtnIcon = QIcon(QPixmap('../res/svg/download.svg'))
        self.saveHoloBtn.setIcon(saveHoloBtnIcon)
        self.saveHoloBtn.setIconSize(QSize(14, 14))
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
        calHoloLayout.addWidget(enableSSIMText, 5, 0, 1, 2)
        calHoloLayout.addWidget(self.enableSSIMChk, 5, 2, 1, 1)
        calHoloLayout.addWidget(enableAmpEncText, 5, 3, 1, 2)
        calHoloLayout.addWidget(self.enableAmpEncChk, 5, 5, 1, 1)
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
        self.toggleCamBtn = QPushButton(' 打开相机')
        self.openCamIcon = QIcon(QPixmap('../res/svg/camera-video-fill.svg'))
        self.closeCamIcon = QIcon(QPixmap('../res/svg/camera-video-off-fill.svg'))
        self.toggleCamBtn.setIcon(self.openCamIcon)
        self.toggleCamBtn.setIconSize(QSize(14, 14))
        self.toggleCamBtn.setStyleSheet("color: #ffffff; background-color: #228B22;")
        self.toggleCamBtn.clicked.connect(self.toggleCam)

        self.snapBtn = QPushButton(' 抓图')
        snapBtnIcon = QIcon(QPixmap('../res/svg/camera2.svg'))
        self.snapBtn.setIcon(snapBtnIcon)
        self.snapBtn.setIconSize(QSize(14, 14))
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

        camCtrlLayout = QGridLayout()
        camCtrlLayout.addWidget(self.toggleCamBtn, 0, 0, 1, 1)
        camCtrlLayout.addWidget(self.snapBtn, 0, 1, 1, 1)
        camCtrlLayout.addWidget(expTimeText, 1, 0, 1, 2)
        camCtrlLayout.addWidget(self.expTimeInput, 2, 0, 1, 1)
        camCtrlLayout.addLayout(autoExpLayout, 2, 1, 1, 1)
        camCtrlLayout.setColumnStretch(0, 1)
        camCtrlLayout.setColumnStretch(1, 1)

        camCtrlGroupBox = QGroupBox("相机设置")
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
        self.targetImgPreview.setText("从输入模块载入已有目标图 \n或打开相机后 [从相机捕获]")
        self.targetImgPreview.setMaximumSize(320, 320)

        targetImgHLayout = QHBoxLayout()
        targetImgHLayout.addWidget(self.targetImgPreview)
        targetImgVLayout = QVBoxLayout()
        targetImgVLayout.addLayout(targetImgHLayout)
        targetImgWidget = QWidget()
        targetImgWidget.setLayout(targetImgVLayout)

        # 全息图显示区域
        self.holoImgPreview = QLabel()
        self.holoImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.holoImgPreview.setText("从输入模块载入已有全息图 \n或从目标图 [计算全息图]")
        self.holoImgPreview.setMaximumSize(320, 320)

        holoImgHLayout = QHBoxLayout()
        holoImgHLayout.addWidget(self.holoImgPreview)
        holoImgVLayout = QVBoxLayout()
        holoImgVLayout.addLayout(holoImgHLayout)
        holoImgWidget = QWidget()
        holoImgWidget.setLayout(holoImgVLayout)

        # 重建光场仿真显示区域
        self.reconstructImgPreview = QLabel()
        self.reconstructImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.reconstructImgPreview.setText(f"计算或载入全息图后查看结果")
        self.reconstructImgPreview.setMaximumSize(320, 320)

        reconstructImgHLayout = QHBoxLayout()
        reconstructImgHLayout.addWidget(self.reconstructImgPreview)
        reconstructImgVLayout = QVBoxLayout()
        reconstructImgVLayout.addLayout(reconstructImgHLayout)
        reconstructImgWidget = QWidget()
        reconstructImgWidget.setLayout(reconstructImgVLayout)

        self.holoImgPreviewTabWidget = QTabWidget()
        self.holoImgPreviewTabWidget.addTab(targetImgWidget, "目标图")
        self.holoImgPreviewTabWidget.addTab(holoImgWidget, "全息图")
        self.holoImgPreviewTabWidget.addTab(reconstructImgWidget, "重建光场")
        self.holoImgPreviewTabWidget.setTabEnabled(2, False)

        # 重建光场相位分布显示区域
        self.reconstructPhase2D = Figure()
        reconstructPhase2DCanvas = FigureCanvas(self.reconstructPhase2D)

        # 迭代历史显示区域
        self.iterationHistory = Figure()
        iterationHistoryCanvas = FigureCanvas(self.iterationHistory)

        self.reconstructViewTabWidget = QTabWidget()
        self.reconstructViewTabWidget.addTab(iterationHistoryCanvas, "迭代历史")
        self.reconstructViewTabWidget.addTab(reconstructPhase2DCanvas, "重建相位")
        self.reconstructViewTabWidget.setTabEnabled(1, False)

        calcInfoLayout = QVBoxLayout()
        calcInfoLayout.addWidget(self.holoImgPreviewTabWidget)
        calcInfoLayout.addWidget(self.reconstructViewTabWidget)
        calcInfoLayout.setStretch(0, 1)
        calcInfoLayout.setStretch(1, 1)

        self.calcInfoWidget = QWidget()
        self.calcInfoWidget.setLayout(calcInfoLayout)

        imgViewAreaLayout = QHBoxLayout()
        imgViewAreaLayout.addWidget(self.calcInfoWidget)
        imgViewAreaLayout.addWidget(camPreviewGroupBox)
        imgViewAreaLayout.setStretch(0, 2)
        imgViewAreaLayout.setStretch(1, 5)

        imgViewArea = QWidget()
        imgViewArea.setObjectName("imgViewArea")
        imgViewArea.setLayout(imgViewAreaLayout)

        # ==== 状态栏 ====

        self.toggleCalcLayoutBtn = QPushButton()
        self.openLayoutIcon = QIcon(QPixmap('../res/svg/fullscreen-exit.svg'))
        self.closeLayoutIcon = QIcon(QPixmap('../res/svg/fullscreen.svg'))
        self.toggleCalcLayoutBtn.setText(" 收起全息图面板")
        self.toggleCalcLayoutBtn.setIcon(self.closeLayoutIcon)
        self.toggleCalcLayoutBtn.setIconSize(QSize(14, 14))
        self.toggleCalcLayoutBtn.clicked.connect(self.toggleCalcLayout)

        self.progressBar = QProgressBar()
        self.progressBar.setStyleSheet("QProgressBar {min-width: 100px; max-width: 200px;}")

        self.secondStatusInfo = QLabel()
        self.secondStatusInfo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.secondStatusInfo.setStyleSheet("QLabel {min-width: 100px}")

        self.statusBar = QStatusBar()
        self.statusBar.addPermanentWidget(self.secondStatusInfo)
        self.statusBar.addPermanentWidget(self.progressBar)
        self.statusBar.addPermanentWidget(self.toggleCalcLayoutBtn)

        self.statusBar.showMessage(f"就绪")

        # ==== 窗体 ====
        layout = QHBoxLayout()
        layout.addWidget(ctrlArea)
        layout.addWidget(imgViewArea)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setStatusBar(self.statusBar)

    def _initLaserUI(self):
        # 激光器功能区
        self.laserPortSel = QComboBox()
        self.laserPortSel.setEnabled(False)

        self.connectLaserBtn = QPushButton(' 连接控制盒')
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

        laserPwrText = QLabel("激光功率 (mW)")

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
        laserCtrlLayout.addWidget(self.laserPortSel, 0, 0, 1, 2)
        laserCtrlLayout.addWidget(self.connectLaserBtn, 1, 0, 1, 1)
        laserCtrlLayout.addWidget(self.toggleLaserBtn, 1, 1, 1, 1)
        laserCtrlLayout.addWidget(laserPwrText, 2, 0, 1, 2)
        laserCtrlLayout.addWidget(self.laserPwrInput, 3, 0, 1, 1)
        laserCtrlLayout.addWidget(self.setLaserPwrBtn, 3, 1, 1, 1)
        laserCtrlLayout.setColumnStretch(0, 1)
        laserCtrlLayout.setColumnStretch(1, 1)

        self.laserCtrlGroupBox = QGroupBox("激光器设置")
        self.laserCtrlGroupBox.setLayout(laserCtrlLayout)

        self.laser = LaserMiddleWare()
        self.deviceUpdatedEvent()

    def toggleCalcLayout(self):
        visible = not self.calcInfoWidget.isVisible()
        self.calcInfoWidget.setVisible(visible)
        if visible:
            self.toggleCalcLayoutBtn.setText(" 收起全息图面板")
            self.toggleCalcLayoutBtn.setIcon(self.closeLayoutIcon)
        else:
            self.toggleCalcLayoutBtn.setText(" 展开全息图面板")
            self.toggleCalcLayoutBtn.setIcon(self.openLayoutIcon)

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
            self.connectLaserBtn.setText(' 连接控制盒')
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
                    self.connectLaserBtn.setText(' 断开控制盒')
                    self.connectLaserBtn.setIcon(self.disconnectLaserIcon)

        self.laserPortSel.setEnabled(not self.laserPortSel.isEnabled())
        self.toggleLaserBtn.setEnabled(not self.laserPortSel.isEnabled())
        if self.toggleLaserBtn.isEnabled():
            self.toggleLaserBtn.setStyleSheet("color: #000000; background-color: #f38d05;")
        else:
            self.toggleLaserBtn.setStyleSheet("")
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

            self.toggleCamBtn.setText(" 打开相机")
            self.toggleCamBtn.setIcon(self.openCamIcon)
            self.toggleCamBtn.setStyleSheet("color: #ffffff; background-color: #228B22;")
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

                self.toggleCamBtn.setText(" 关闭相机")
                self.toggleCamBtn.setIcon(self.closeCamIcon)
                self.toggleCamBtn.setStyleSheet("color: #000000; background-color: #f38d05;")
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
        todo: SSIE, PSNR, 直方图
        """
        # 已载入图像
        if self.pendingImg is not None:
            self.targetImg = self.pendingImg

            logHandler.info(f"Start Calculation.")
            self.statusBar.showMessage(f"开始计算...")
            self.progressBar.reset()
            self.progressBar.setRange(0, 10)
            self.secondStatusInfo.setText(f"归一化与类型转换...")
            self.progressBar.setValue(1)

            self._uniList.clear()
            self._effiList.clear()
            self._RMSEList.clear()
            self._SSIMList.clear()

            maxIterNum = self.maxIterNumInput.value()
            iterTarget = self.iterTargetInput.value() * 0.01

            targetNormalized = self.targetImg / 255

            # CuPy类型转换 (NumPy->CuPy)
            target = cp.asarray(targetNormalized)

            # 计时
            self.secondStatusInfo.setText(f"创建计算实例...")
            self.progressBar.setValue(2)
            tStart = time.time()
            try:
                if self.holoAlgmSel.currentIndex() == 0:
                    algorithm = WCIA(
                        target, maxIterNum,
                        initPhase=(self.initPhaseSel.currentIndex(), None),
                        iterTarget=(self.iterTargetSel.currentIndex(), iterTarget),
                        enableSSIM=self.enableSSIMChk.isChecked(),
                        uniList=self._uniList,
                        effiList=self._effiList,
                        RMSEList=self._RMSEList,
                        SSIMList=self._SSIMList
                    )
                elif self.holoAlgmSel.currentIndex() == 1:
                    algorithm = GSW(
                        target, maxIterNum,
                        initPhase=(self.initPhaseSel.currentIndex(), None),
                        iterTarget=(self.iterTargetSel.currentIndex(), iterTarget),
                        enableSSIM=self.enableSSIMChk.isChecked(),
                        uniList=self._uniList,
                        effiList=self._effiList,
                        RMSEList=self._RMSEList,
                        SSIMList=self._SSIMList
                    )

                self.secondStatusInfo.setText(f"开始迭代...")
                self.progressBar.setRange(0, 0)

                u, phase = algorithm.iterate()
                if self.enableAmpEncChk.isChecked():
                    u, phase = Holo.encodeAmp2Phase(u)

            except Exception as err:
                logHandler.error(f"Err in iteration: {err}")
                QMessageBox.critical(self, '错误', f'迭代过程中发生异常：\n{err}')
                self.statusBar.showMessage(f"迭代过程中发生异常: {err}")
            else:
                self.secondStatusInfo.setText(f"类型转换...")
                self.progressBar.setRange(0, 10)
                self.progressBar.setValue(5)
                # CuPy类型转换 (CuPy->NumPy)
                self.holoImg = cp.asnumpy(Holo.genHologram(phase))
                self.holoU = cp.asnumpy(u)

                self.holoImgRotated = cv2.flip(self.holoImg, 1)

                tEnd = time.time()

                del algorithm

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
                duration = round(tEnd - tStart, 2)
                uniformity = self._uniList[-1]
                efficiency = self._effiList[-1]
                RMSE = self._RMSEList[-1]
                if self.enableSSIMChk.isChecked():
                    SSIM = self._SSIMList[-1]
                else:
                    SSIM = -1

                self.iterationDraw()

                self.secondStatusInfo.setText(f"完成")
                self.progressBar.setValue(10)
                logHandler.info(f"Finish Calculation.")
                logHandler.info(
                    f"Iteration={iteration}, Duration={duration}s, "
                    f"uniformity={uniformity}, efficiency={efficiency}, RMSE={RMSE}, SSIM={SSIM}"
                )
                self.statusBar.showMessage(
                    f"计算完成。迭代{iteration}次，时长 {duration}s，"
                    f"均匀度{round(uniformity, 4)}，光场利用效率{round(efficiency, 4)}, "
                    f"RMSE={round(RMSE, 4)}, SSIM={round(SSIM, 4)}"
                )

                self.holoImgPreviewTabWidget.setCurrentIndex(2)
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

            self.pendingImg = ImgProcess().createGaussianBeamMask(self.pendingImg, 6)

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

                self.holoImgPreviewTabWidget.setCurrentIndex(0)
                self.holoImgPreviewTabWidget.setTabEnabled(2, False)
                self.holoImgPreview.setText("从输入模块载入已有全息图 \n或从目标图 [计算全息图]")
                self.reconstructImgPreview.setText(f"计算或载入全息图后查看结果")

                self.calcHoloBtn.setEnabled(True)
                self.holoAlgmSel.setEnabled(True)
                self.initPhaseSel.setEnabled(True)
                self.maxIterNumInput.setEnabled(True)
                self.iterTargetSel.setEnabled(True)
                self.iterTargetInput.setEnabled(True)
                self.enableSSIMChk.setEnabled(True)
                self.enableAmpEncChk.setEnabled(True)
                self.reconstructViewTabWidget.setCurrentIndex(0)
                self.reconstructViewTabWidget.setTabEnabled(1, False)

                self.iterationHistory.clf()
                self.iterationHistory.canvas.draw()

            self.saveHoloBtn.setEnabled(False)
            logHandler.debug(f"UI thread updated. Initiator=User  Mode=refresh")
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

                self.holoImgPreviewTabWidget.setCurrentIndex(1)
                self.holoImgPreviewTabWidget.setTabEnabled(2, True)
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
                self.enableSSIMChk.setEnabled(False)
                self.enableAmpEncChk.setEnabled(True)
                self.reconstructViewTabWidget.setCurrentIndex(0)
                self.reconstructViewTabWidget.setTabEnabled(1, True)

                self.iterationHistory.clf()
                self.iterationHistory.canvas.draw()
        else:
            logHandler.warning(f"No image loaded. ")

    def reconstructResult(self, holoU, d: int, wavelength: float):
        """
        [UI事件] 重建光场和相位
        :param holoU: 全息光场
        :param d: 像面距离
        :param wavelength: 激光波长
        """

        def onMousePress2D(event):
            """
            2D相位重建 鼠标按下事件
            """
            if event.inaxes:  # 判断鼠标是否在axes内
                if event.button == 1:  # 判断按下的是否为鼠标左键1（右键是3）
                    self._pressed = True
                    self._lastX = event.xdata  # 获取鼠标按下时的坐标X
                    self._lastY = event.ydata  # 获取鼠标按下时的坐标Y

        def onMouseMove2D(event):
            """
            2D相位重建 鼠标移动事件
            """
            if event.inaxes:
                if self._pressed:  # 按下状态
                    # 计算新的坐标原点并移动
                    # 获取当前最新鼠标坐标与按下时坐标的差值

                    xDelta = event.xdata - self._lastX
                    yDelta = event.ydata - self._lastY
                    # 获取当前原点和最大点的4个位置
                    xMin, xMax = event.inaxes.get_xlim()
                    yMin, yMax = event.inaxes.get_ylim()

                    xMin = xMin - xDelta
                    xMax = xMax - xDelta
                    yMin = yMin - yDelta
                    yMax = yMax - yDelta

                    event.inaxes.set_xlim(xMin, xMax)
                    event.inaxes.set_ylim(yMin, yMax)
                    self.reconstructPhase2D.canvas.draw()

        def onMouseRelease2D(event):
            """
            2D相位重建 鼠标离开事件
            """
            if self._pressed:
                self._pressed = False  # 鼠标松开，结束移动

        def onScroll2D(event):
            """
            2D相位重建 鼠标滚动事件
            """
            xMin, xMax = event.inaxes.get_xlim()
            yMin, yMax = event.inaxes.get_ylim()
            xRange = (xMax - xMin) / 10
            yRange = (yMax - yMin) / 10
            if event.button == 'up':
                event.inaxes.set(xlim=(xMin + xRange, xMax - xRange))
                event.inaxes.set(ylim=(yMin + yRange, yMax - yRange))
            elif event.button == 'down':
                event.inaxes.set(xlim=(xMin - xRange, xMax + xRange))
                event.inaxes.set(ylim=(yMin - yRange, yMax + yRange))

            self.reconstructPhase2D.canvas.draw()

        self.holoImgPreviewTabWidget.setTabEnabled(2, True)
        self.reconstructViewTabWidget.setTabEnabled(1, True)

        # 重建光场
        reconstructU = Holo.reconstruct(holoU, d, wavelength)
        reconstructA = Holo.normalize(np.abs(reconstructU)) * 255
        reconstructP = np.where(np.angle(reconstructU) < 0, np.angle(reconstructU) + 2 * np.pi, np.angle(reconstructU))

        ImgProcess.cvImg2QPixmap(self.reconstructImgPreview, reconstructA.astype("uint8"))

        # 重建相位
        self.reconstructPhase2D.clf()
        self.reconstructPhase2D.canvas.mpl_connect('scroll_event', onScroll2D)
        self.reconstructPhase2D.canvas.mpl_connect("button_press_event", onMousePress2D)
        self.reconstructPhase2D.canvas.mpl_connect("button_release_event", onMouseRelease2D)
        self.reconstructPhase2D.canvas.mpl_connect("motion_notify_event", onMouseMove2D)

        phasePlot = self.reconstructPhase2D.add_subplot(111)

        p = phasePlot.imshow(reconstructP, cmap='RdYlGn')

        phasePlot.set_xticks([])
        phasePlot.set_yticks([])

        cb = self.reconstructPhase2D.colorbar(
            p, ax=phasePlot, orientation='horizontal', fraction=0.1, pad=0.03)
        cb.set_ticks([0, np.pi/2, np.pi, 3 * np.pi/2, 2 * np.pi])
        cb.formatter = ticker.FixedFormatter(['0', 'π/2', 'π', '3π/2', '2π'])
        cb.ax.tick_params(labelsize=8)

        phasePlot.spines['right'].set_visible(False)
        phasePlot.spines['top'].set_visible(False)
        phasePlot.spines['left'].set_visible(False)
        phasePlot.spines['bottom'].set_visible(False)

        self.reconstructPhase2D.tight_layout()
        self.reconstructPhase2D.canvas.draw()

    def iterationDraw(self):
        """
        [UI事件] 迭代参数分析
        """
        def figureHoverEventIter(event):
            # 当鼠标在图表上移动时，获取当前位置的x坐标值
            x = event.xdata
            mouseX, mouseY = iterationPlot.transData.inverted().transform((event.x, event.y))

            # 如果x在数据范围内，则找到对应的y坐标值
            if x is not None and 0 <= x <= iterationPlot.get_xlim()[1]:
                x_index = np.searchsorted(xData, x, side='left')
                if 0 < x_index <= len(xData):
                    # 获取最接近的x坐标的数据点插值结果
                    xVal = xData[x_index - 1]
                    yUni = self._uniList[x_index - 1]
                    yEffi = self._effiList[x_index - 1]
                    yRMSE = self._RMSEList[x_index - 1]
                    if self.enableSSIMChk.isChecked():
                        ySSIM = self._SSIMList[x_index - 1]
                        ssimTxt = f"\nSSIM={ySSIM:.4f}"
                    else:
                        ssimTxt = ""

                    # 格式化坐标数据
                    xyText = f"X={xVal:}\nUni={yUni:.4f}\nEffi={yEffi:.4f}\nRMSE={yRMSE:.4f}{ssimTxt}"

                    # 更新文本对象的内容
                    iterText.set_text(xyText)
                    iterText.set_position(
                        (mouseX - 0.1 * iterationPlot.get_xlim()[1],
                         mouseY + 0.2 * iterationPlot.get_ylim()[1])
                    )
                    iterText.set_visible(True)  # 显示文本

                    # 显示垂线
                    vline.set_data([xVal], [iterationPlot.get_ylim()[0], iterationPlot.get_ylim()[1]])
                    vline.set_visible(True)

                    # 计算并显示交点
                    crossPtU.set_data([xVal, xVal], [yUni, yUni])
                    crossPtU.set_visible(True)

                    crossPtE.set_data([xVal, xVal], [yEffi, yEffi])
                    crossPtE.set_visible(True)

                    crossPtR.set_data([xVal, xVal], [yRMSE, yRMSE])
                    crossPtR.set_visible(True)

                    if self.enableSSIMChk.isChecked():
                        crossPtS.set_data([xVal, xVal], [ySSIM, ySSIM])
                        crossPtS.set_visible(True)

                    self.iterationHistory.canvas.draw()  # 重绘画布

                    # 更新状态栏的显示
                    self.secondStatusInfo.setText(xyText.replace('\n', ', '))
                else:
                    # 如果x不在数据范围内，则隐藏文本对象
                    iterText.set_visible(False)
                    vline.set_visible(False)
                    crossPtU.set_visible(False)
                    crossPtE.set_visible(False)
                    crossPtR.set_visible(False)
                    if self.enableSSIMChk.isChecked():
                        crossPtS.set_visible(False)
                    self.iterationHistory.canvas.draw()
            else:
                # 如果x不在数据范围内，则隐藏文本对象
                iterText.set_visible(False)
                vline.set_visible(False)
                crossPtU.set_visible(False)
                crossPtE.set_visible(False)
                crossPtR.set_visible(False)
                if self.enableSSIMChk.isChecked():
                    crossPtS.set_visible(False)
                self.iterationHistory.canvas.draw()

        self.iterationHistory.clf()
        self.iterationHistory.canvas.mpl_connect('motion_notify_event', figureHoverEventIter)

        iterationPlot = self.iterationHistory.add_subplot(111)

        xData = np.arange(1, len(self._uniList) + 1)

        iterationPlot.plot(xData, self._uniList, label='Uniformity')
        iterationPlot.plot(xData, self._effiList, label='efficiency')
        iterationPlot.plot(xData, self._RMSEList, label='RMSE')
        if self.enableSSIMChk.isChecked():
            iterationPlot.plot(xData, self._SSIMList, label='SSIM')

        iterationPlot.legend(loc='best', fontsize=8)

        iterationPlot.set_xlabel("Iteration", fontsize=8)
        iterationPlot.tick_params(axis='x', labelsize=8)
        iterationPlot.tick_params(axis='y', labelsize=8)

        # 创建一个用于显示坐标的文本对象，初始时不显示
        iterText = iterationPlot.text(
            0, 0, '', transform=iterationPlot.transData,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        iterText.set_visible(False)  # 初始时不显示文本

        # 创建一个垂线对象
        vline, = iterationPlot.plot([], [], '-', color='silver', lw=0.5)
        vline.set_visible(False)  # 初始时不显示垂线

        # 创建一个用于显示交点的小圆点对象
        crossPtU = iterationPlot.plot([], [], 'o', color='C0', markersize=5)[0]
        crossPtU.set_visible(False)

        crossPtE = iterationPlot.plot([], [], 'o', color='C1', markersize=5)[0]
        crossPtE.set_visible(False)

        crossPtR = iterationPlot.plot([], [], 'o', color='C2', markersize=5)[0]
        crossPtR.set_visible(False)

        if self.enableSSIMChk.isChecked():
            crossPtS = iterationPlot.plot([], [], 'o', color='C3', markersize=5)[0]
            crossPtS.set_visible(False)

        self.iterationHistory.tight_layout()
        self.iterationHistory.subplots_adjust(left=0.1, bottom=0.12)
        self.iterationHistory.canvas.draw()

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
            autodetect = any(arg == '-d' or arg == '--auto-detect' for arg in sys.argv[1:])
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
                        f'程序被配置为[自动检测LCOS设备]模式，但未检测到名为 "JDC EDK" 的LCOS硬件。\n'
                        f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                    )
                    logHandler.error(
                        f"Auto detect mode, No LCOS named 'JDC EDK' detected. "
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
    args = Utils.getCmdOpt()

    logHandler = Utils.getLog(
        consoleLevel=args.cloglvl,
        fileLevel=args.floglvl,
        writeLogFile=(args.floglvl > 0)
    )

    # QApplication.setStyle(QStyleFactory.create('Fusion'))
    app = QApplication(sys.argv)
    qInstallMessageHandler(Utils.exceptionHandler)
    window = MainWindow()
    window.showMaximized()

    sys.excepthook = Utils.exceptHook
    sys.exit(app.exec())
