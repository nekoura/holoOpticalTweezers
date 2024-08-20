import sys
import ctypes
import time
import cv2
import cupy as cp
import numpy as np
from pathlib import Path
from queue import Queue
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal, pyqtSlot, QSize, QRect, qInstallMessageHandler, QTimer
from PyQt6.QtGui import QGuiApplication, QPixmap, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, \
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, \
    QFileDialog, QMessageBox, \
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar
from lib.utils.utils import Utils, ImgProcess
from lib.utils.autoDetect import FeaturesDetect, FeaturesSort, \
    FrameGeneratorWorker, HoloGeneratorWorker
from lib.holo.libHoloAlgmGPU import WCIA
from lib.holo.libHoloEssential import Holo, HoloCalcWorker
from lib.cam.camAPI import CameraMiddleware
from multiprocessing import Pipe

class MainWindow(QMainWindow):
    """
    [主窗口类] 包含预览与控制面板

    :var snapImg: 现场的图像
    :var targetImg: 全息图目标图像
    :var holoU: 全息图复光场
    :var holoImg: 全息图(相位项转位图)
    :var holoImgRotated: 针对LCOS旋转方向的全息图
    :var zerothOrderPosition: 激光零级位置
    """
    holoImgReady = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.snapImg = None
        self.targetImg = None
        self.holoImg = None
        self.holoU = None
        self.holoImgRotated = None
        self.zerothOrderPosition = (844, 674)
        self._frameGenerator = None
        self._holoGenerator = None
        self._imgSaver = None
        self._framePipeReceiver, self._framePipeSender = Pipe()
        self._holoPipeReceiver, self._holoPipeSender = Pipe()
        self._imgQueue = Queue()
        self._imgPlayTimer = QTimer()
        self._imgPlayTimer.timeout.connect(self.showNext)
        self._snapAsTarget = False
        self._snapIsSave = False
        self._snapIsDisp = False
        self._uniList = []
        self._effiList = []
        self._RMSEList = []

        # 相机实例通信
        self.cam = CameraMiddleware()
        self.cam.frameUpdate.connect(self.frameRefreshEvent)
        self.cam.snapUpdate.connect(self.snapEvent)
        self.cam.expoUpdate.connect(self.expTimeUpdatedEvent)

        # 副屏实例通信
        self.secondWin = SecondMonitorWindow()
        self.holoImgReady.connect(self.secondWin.displayHoloImg)

        self._initUI()

        self.toggleCam()
        logHandler.info("Camera started automatically.")

    def _initUI(self):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SLM Holograph Generator")
        self.setWindowFlags(Qt.WindowType.WindowCloseButtonHint)
        self.setWindowIcon(QIcon("../res/slm.ico"))
        self.setWindowTitle("SLM Holograph Generator")
        self.resize(1200, 600)

        # ==== 控制区域 ====

        # 相机功能区
        self.snapBtn = QPushButton('抓图')
        self.snapBtn.clicked.connect(self.snapAndSave)
        self.snapBtn.setEnabled(False)

        expTimeText = QLabel("曝光 (ms)")

        self.expTimeInput = QSpinBox()
        self.expTimeInput.setEnabled(False)
        self.expTimeInput.textChanged.connect(self.expTimeSet)

        camCtrlLayout = QGridLayout()
        camCtrlLayout.addWidget(expTimeText, 0, 0, 1, 2)
        camCtrlLayout.addWidget(self.expTimeInput, 0, 2, 1, 2)
        camCtrlLayout.addWidget(self.snapBtn, 0, 4, 1, 2)
        camCtrlLayout.setColumnStretch(0, 1)
        camCtrlLayout.setColumnStretch(1, 1)
        camCtrlLayout.setColumnStretch(2, 1)
        camCtrlLayout.setColumnStretch(3, 1)
        camCtrlLayout.setColumnStretch(4, 1)
        camCtrlLayout.setColumnStretch(5, 1)

        camCtrlGroupBox = QGroupBox("相机")
        camCtrlGroupBox.setLayout(camCtrlLayout)

        # 计算参数功能区
        holoAlgmText = QLabel("全息算法")

        self.holoAlgmSel = QComboBox()
        self.holoAlgmSel.addItem(f"WCIA")
        self.holoAlgmSel.setEnabled(False)

        initPhaseText = QLabel("初始相位")

        self.initPhaseSel = QComboBox()
        self.initPhaseSel.addItem(f"随机")
        self.initPhaseSel.addItem(f"目标光场IFFT")
        self.initPhaseSel.setEnabled(False)

        maxIterNumText = QLabel("最大迭代")

        self.maxIterNumInput = QSpinBox()
        self.maxIterNumInput.setRange(0, 10000)
        self.maxIterNumInput.setValue(40)
        self.maxIterNumInput.setEnabled(False)

        iterTargetText = QLabel("终止迭代")
        iterTargetText2 = QLabel("RMSE(%) ≤")

        self.iterTargetInput = QDoubleSpinBox()
        self.iterTargetInput.setRange(0, 100)
        self.iterTargetInput.setValue(1)
        self.iterTargetInput.setEnabled(False)

        holoSetLayout = QGridLayout()
        holoSetLayout.addWidget(holoAlgmText, 0, 0, 1, 2)
        holoSetLayout.addWidget(self.holoAlgmSel, 0, 2, 1, 4)
        holoSetLayout.addWidget(initPhaseText, 1, 0, 1, 2)
        holoSetLayout.addWidget(self.initPhaseSel, 1, 2, 1, 4)
        holoSetLayout.addWidget(maxIterNumText, 2, 0, 1, 2)
        holoSetLayout.addWidget(self.maxIterNumInput, 2, 2, 1, 4)
        holoSetLayout.addWidget(iterTargetText, 3, 0, 1, 2)
        holoSetLayout.addWidget(iterTargetText2, 3, 2, 1, 2)
        holoSetLayout.addWidget(self.iterTargetInput, 3, 4, 1, 2)
        holoSetLayout.setColumnStretch(0, 1)
        holoSetLayout.setColumnStretch(1, 1)
        holoSetLayout.setColumnStretch(2, 1)
        holoSetLayout.setColumnStretch(3, 1)
        holoSetLayout.setColumnStretch(4, 1)
        holoSetLayout.setColumnStretch(5, 1)

        holoSetGroupBox = QGroupBox("计算参数")
        holoSetGroupBox.setLayout(holoSetLayout)

        # 输入功能区
        openTargetFileBtn = QPushButton('打开目标图...')
        openTargetFileBtn.clicked.connect(self.openTargetImg)

        openHoloFileBtn = QPushButton('已有全息图...')
        openHoloFileBtn.clicked.connect(self.openHoloImg)

        inputTipsText = QLabel("欲载入已有全息图，相同目录下需存在保存全息图时生成的同名.npy文件")
        inputTipsText.setStyleSheet("color:#999")
        inputTipsText.setWordWrap(True)

        singleCalcText = QLabel("单次计算")

        self.calcHoloBtn = QPushButton('计算全息图')
        self.calcHoloBtn.clicked.connect(self.calcHoloImg)
        self.calcHoloBtn.setEnabled(False)

        self.saveHoloBtn = QPushButton('保存全息图...')
        self.saveHoloBtn.clicked.connect(self.saveHoloImg)
        self.saveHoloBtn.setEnabled(False)

        autoCalcText = QLabel("自动计算")

        self.autoCalcBtn = QPushButton('从相机捕获')
        self.autoCalcBtn.clicked.connect(self.autoCalcHoloImg)
        self.autoCalcBtn.setEnabled(False)

        calcLayout = QGridLayout()
        calcLayout.addWidget(openTargetFileBtn, 0, 0, 1, 1)
        calcLayout.addWidget(openHoloFileBtn, 0, 1, 1, 1)
        calcLayout.addWidget(inputTipsText, 1, 0, 1, 2)
        calcLayout.addWidget(singleCalcText, 2, 0, 1, 2)
        calcLayout.addWidget(self.calcHoloBtn, 3, 0, 1, 1)
        calcLayout.addWidget(self.saveHoloBtn, 3, 1, 1, 1)
        calcLayout.addWidget(autoCalcText, 4, 0, 1, 2)
        calcLayout.addWidget(self.autoCalcBtn, 5, 0, 1, 2)
        calcLayout.setColumnStretch(0, 1)
        calcLayout.setColumnStretch(1, 1)

        calcGroupBox = QGroupBox("计算")
        calcGroupBox.setLayout(calcLayout)

        # 目标图显示区域
        self.targetImgPreview = QLabel()
        self.targetImgPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.targetImgPreview.setText("从输入模块载入已有目标图 \n或 [从相机捕获]")
        self.targetImgPreview.setFixedSize(240, 240)

        targetImgHLayout = QHBoxLayout()
        targetImgHLayout.addWidget(self.targetImgPreview)
        targetImgVLayout = QVBoxLayout()
        targetImgVLayout.addLayout(targetImgHLayout)
        targetImgGroupBox = QGroupBox("目标图 / 重建预览")
        targetImgGroupBox.setLayout(targetImgVLayout)

        ctrlAreaLayout = QVBoxLayout()
        ctrlAreaLayout.addWidget(camCtrlGroupBox)
        ctrlAreaLayout.addWidget(holoSetGroupBox)
        ctrlAreaLayout.addWidget(calcGroupBox)
        ctrlAreaLayout.addStretch(1)
        ctrlAreaLayout.addWidget(targetImgGroupBox)

        ctrlArea = QWidget()
        ctrlArea.setLayout(ctrlAreaLayout)

        # ==== 显示区域 ====
        # 相机预览区域
        self.camPreview = QLabel()
        self.camPreview.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.camPreview.setMinimumSize(480, 360)

        camPreviewHLayout = QHBoxLayout()
        camPreviewHLayout.addWidget(self.camPreview)
        camPreviewVLayout = QVBoxLayout()
        camPreviewVLayout.addLayout(camPreviewHLayout)
        camPreviewGroupBox = QGroupBox("相机预览")
        camPreviewGroupBox.setLayout(camPreviewVLayout)

        # ==== 状态栏 ====

        self.progressBar = QProgressBar()
        self.progressBar.setStyleSheet("QProgressBar {min-width: 100px; max-width: 300px; margin-right:5px}")
        self.progressBar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.secondStatusInfo = QLabel()
        self.secondStatusInfo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.secondStatusInfo.setStyleSheet("QLabel {min-width: 100px; margin-right:5px}")

        self.setZerothOrderBtn = QPushButton(" 校准 SLM 中心坐标 ")
        self.setZerothOrderBtn.clicked.connect(self.setZerothOrder)
        self.setZerothOrderBtn.setEnabled(False)

        self.statusBar = QStatusBar()
        self.statusBar.addPermanentWidget(self.secondStatusInfo)
        self.statusBar.addPermanentWidget(self.progressBar)
        self.statusBar.addPermanentWidget(self.setZerothOrderBtn)

        self.statusBar.showMessage(f"就绪")
        self.secondStatusInfo.setText(f"等待输入")
        self.progressBar.setRange(0, 10)
        self.progressBar.setValue(0)

        # ==== 窗体 ====

        layout = QHBoxLayout()
        layout.addWidget(ctrlArea)
        layout.addWidget(camPreviewGroupBox)
        layout.setStretch(0, 1)
        layout.setStretch(1, 5)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setStatusBar(self.statusBar)

    def closeEvent(self, event):
        """
        [UI事件] 窗口关闭时同时关闭其他窗口，关闭相机
        """
        for widget in QApplication.instance().allWidgets():
            if isinstance(widget, QWidget) and widget != self:
                widget.close()

        self.cam.closeCamera()
        self.stopThreads()
        logHandler.info(f"Bye.")
        event.accept()

    def toggleCam(self):
        """
        [UI操作] 点击打开相机
        """
        if self.cam.device:
            self.cam.closeCamera()

            logHandler.info(f"Camara closed.")
            self.statusBar.showMessage(f"相机已停止")
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

                expMin, expMax, expCur = self.cam.device.get_ExpTimeRange()
                self.expTimeInput.setRange(10, 1000)
                self.expTimeInput.setValue(int(expCur / 1000))

                self.cam.expoUpdateEvt()

        self.setZerothOrderBtn.setEnabled(not self.setZerothOrderBtn.isEnabled())
        self.snapBtn.setEnabled(not self.snapBtn.isEnabled())
        self.expTimeInput.setEnabled(not self.expTimeInput.isEnabled())

    def snapAndSave(self):
        """
        [UI操作] 点击抓图
        """
        if self.cam.device:
            self.cam.device.Snap(self.cam.RESOLUTION)
            self._snapIsSave = True
            self._snapAsTarget = False
            self._snapIsDisp = False

    def snapAsTarget(self, isSave):
        """
        [UI操作] 点击从相机捕获
        """
        if self.cam.device:
            self.cam.device.Snap(self.cam.RESOLUTION)
            self._snapAsTarget = True
            self._snapIsSave = isSave
            self._snapIsDisp = False

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
            imgDir, _ = QFileDialog.getOpenFileName(
                self,
                "打开图片",
                "",
                "图像文件(*.jpg *.png *.tif *.bmp)",
            )
        except Exception as err:
            QMessageBox.critical(self, '错误', f'载入图像失败：\n{err}')
            logHandler.error(f"Fail to load image: {err}")
        else:
            if imgDir is not None and imgDir != '':
                try:
                    self.targetImg = ImgProcess.loadImg(imgDir)
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
        if self.targetImg is not None:

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
        """
        [UI事件] 全息图计算结果后处理
        """
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

        self.holoImgReady.emit(self.holoImgRotated)
        logHandler.info(f"Image has been transferred to the second monitor.")

        self.secondStatusInfo.setText(f"计算重建效果...")
        self.progressBar.setValue(8)
        self.reconstructResult(self.holoU, 50, 532e-6)

        self.secondStatusInfo.setText(f"进行性能统计...")
        self.progressBar.setValue(9)

        # 性能估计
        iteration = len(self._RMSEList)
        duration = round(tEnd - self.tStart, 2)
        RMSE = self._RMSEList[-1]

        self.secondStatusInfo.setText(f"完成")
        self.progressBar.setValue(10)
        logHandler.info(f"Finish Calculation.")
        self.statusBar.showMessage(
            f"计算完成。迭代{iteration}次，时长 {duration}s，"
            f"RMSE={round(RMSE, 4)}"
        )

    def snapEvent(self):
        """
        [UI事件] 抓图
        """
        if self._snapAsTarget:
            self.snapImg = cv2.cvtColor(self.cam.snapshot, cv2.COLOR_RGB2GRAY)

            if self._snapIsDisp:
                self.imgLoadedEvent('Snap', None)
            self._snapAsTarget = False

            filepath = f"../pics/snap/{time.strftime('%Y%m%d%H%M%S')}-asTarget.jpg"
        else:
            filepath = f"../pics/snap/{time.strftime('%Y%m%d%H%M%S')}.jpg"

        Utils.folderPathCheck(filepath)

        if self._snapIsSave:
            cv2.imwrite(f"{filepath}", self.cam.snapshot)
            logHandler.info(f"Snapshot saved as '{filepath}'")
            self.statusBar.showMessage(f"截图已保存至 '{filepath}'")

    def frameRefreshEvent(self):
        """
        [UI事件] 刷新相机预览窗口
        """
        preview = self.cam.frame.scaled(
            self.camPreview.width(), self.camPreview.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        )
        self.camPreview.setPixmap(QPixmap.fromImage(preview))

    def imgLoadedEvent(self, targetDir, holoDir):
        """
        [UI事件] 载入图像后界面刷新
        """
        if targetDir is not None and targetDir != '':  # 载入目标图
            if self.targetImg is not None:
                imgRes = f"{self.targetImg.shape[1]}x{self.targetImg.shape[0]}"
                logHandler.info(
                    f"Image Loaded. Dir={targetDir}, Resolution={imgRes}"
                )

                self.statusBar.showMessage(f"已加载图像{targetDir}，分辨率{imgRes}")
                self.holoImg = None
                ImgProcess.cvImg2QPixmap(self.targetImgPreview, self.targetImg)

                self.calcHoloBtn.setEnabled(True)
                self.holoAlgmSel.setEnabled(True)
                self.initPhaseSel.setEnabled(True)
                self.maxIterNumInput.setEnabled(True)
                self.iterTargetInput.setEnabled(True)
                self.autoCalcBtn.setEnabled(True)
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
                self.targetImg = None
                self.reconstructResult(self.holoU, 50, 532e-6)

                self.holoImgRotated = cv2.rotate(self.holoImg, cv2.ROTATE_90_CLOCKWISE)
                self.holoImgReady.emit(self.holoImgRotated)

                self.calcHoloBtn.setEnabled(False)
                self.holoAlgmSel.setEnabled(False)
                self.initPhaseSel.setEnabled(False)
                self.maxIterNumInput.setEnabled(False)
                self.iterTargetInput.setEnabled(False)
                self.autoCalcBtn.setEnabled(False)
        else:
            logHandler.warning(f"No image loaded. ")

    def showNext(self):
        try:
            (holoImg, index) = self._imgQueue.get()
        except Exception as e:
            print(e)
        else:
            if self._imgQueue.qsize() == 0:
                self._imgPlayTimer.stop()
                self._framePipeReceiver = None
                self._framePipeSender = None
                self._holoPipeReceiver = None
                self._holoPipeSender = None
                self.statusBar.showMessage(f"显示完成")
                self.autoCalcBtn.setText("从相机捕获")
                self.autoCalcBtn.clicked.disconnect()
                self.autoCalcBtn.clicked.connect(self.autoCalcHoloImg)
            else:
                self.holoImgReady.emit(holoImg)
                self.statusBar.showMessage(f"正在显示第{index}帧")
                QApplication.processEvents()

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

        ImgProcess.cvImg2QPixmap(self.targetImgPreview, reconstructA.astype("uint8"))

    def expTimeUpdatedEvent(self):
        """
        [UI事件] 处理相机更新曝光时间状态
        """
        expTime = self.cam.device.get_ExpoTime()
        # 当相机回调参数时阻止用户侧输入
        with QSignalBlocker(self.expTimeInput):
            self.expTimeInput.setValue(int(expTime / 1000))

    def setZerothOrder(self):
        message = QMessageBox.warning(
            self,
            '校准 SLM 中心坐标',
            f'校准 SLM 中心坐标有助于匹配视场和全息图之间的位置关系。在调整光路后，务必校准 SLM 的中心坐标。\n\n'
            f'校准前请完成以下准备步骤：\n'
            f'1. 取下滤光片，放置反射镜，确认物镜倍率为50x及以上；\n'
            f'2. 关闭明场灯，确认SLM未加载任何图像；\n'
            f'3. 打开激光，调整焦距、功率和曝光时间，使零级光清晰可见。\n'
            f'将零级光位置调节得尽量靠近视场中心，有助于最大化成像区域有效面积。\n\n'
            f'完成上述步骤后点击 [OK]，进入下一步。',
            (QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok),
            QMessageBox.StandardButton.Cancel
        )

        if message == QMessageBox.StandardButton.Ok:
            self.snapAsTarget(False)
            QMessageBox.warning(
                self,
                '准备识别',
                f'程序即将开始识别中心坐标。识别期间请保持平台稳定，勿操作平台。\n'
                f'准备好后点击 [OK]'
            )
            if self.snapImg is not None:
                try:
                    circles = FeaturesDetect.detectCircles(self.snapImg)
                except TypeError:
                    QMessageBox.critical(
                        self,
                        '错误',
                        f'未识别到零级光斑'
                    )
                    return -1
                else:
                    self.zerothOrderPosition = circles[0][:2]
                    QMessageBox.information(
                        self,
                        '成功',
                        f'中心坐标更新为{self.zerothOrderPosition}'
                    )
                    logHandler.info(f"zerothOrderPosition at {self.zerothOrderPosition}")
                    return 0
            else:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未识别到零级光斑'
                )
                return -1

    def autoCalcHoloImg(self):

        def startThreads():
            self._frameGenerator.start()
            self._holoGenerator.start()

            self._imgPlayTimer.start(50)

            self._framePipeSender.close()
            self._framePipeReceiver.close()
            self._holoPipeSender.close()

            while True:
                try:
                    (phase, index) = self._holoPipeReceiver.recv()
                    holoImg = cp.asnumpy(Holo.genHologram(phase))
                    holoImgRotated = cv2.rotate(holoImg, cv2.ROTATE_90_CLOCKWISE)
                    self._imgQueue.put((holoImgRotated, index))
                    self.secondStatusInfo.setText(f"计算第{index}帧")
                    QApplication.processEvents()
                except EOFError:
                    self._holoPipeReceiver.close()
                    self.progressBar.setRange(0, 100)
                    self.progressBar.setValue(100)
                    self.secondStatusInfo.setText(f"计算已完成")
                    break

        self.snapAsTarget(False)

        message = QMessageBox.warning(
            self,
            '准备识别',
            f'程序即将开始识别图像中目标位置。识别期间请保持平台稳定，勿操作平台。\n'
            f'准备好后点击 [OK]',
            (QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok),
            QMessageBox.StandardButton.Ok
        )

        if message == QMessageBox.StandardButton.Ok:
            if all([self._framePipeReceiver, self._framePipeSender, self._holoPipeReceiver, self._holoPipeSender]):
                pass
            else:
                self._framePipeReceiver, self._framePipeSender = Pipe()
                self._holoPipeReceiver, self._holoPipeSender = Pipe()

            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
            self.autoCalcBtn.setText("中止")
            self.autoCalcBtn.clicked.disconnect()
            self.autoCalcBtn.clicked.connect(self.stopThreads)

            center = self.zerothOrderPosition

            currentImg = FeaturesDetect.cutImg(self.snapImg, center)
            # currentImg = FeaturesDetect.cutImg(cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE), center)

            # 检测特征点
            try:
                currentPoints = FeaturesDetect.detectCircles(currentImg)
                targetPoints = FeaturesDetect.detectCircles(self.targetImg)

            except TypeError:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未能识别到目标点或图像点\n'
                )
                self.stopThreads()
                return -1

            self.secondStatusInfo.setText("识别目标点...")
            totalOrderA, _ = FeaturesSort(targetPoints, 0.1).calc()

            self.secondStatusInfo.setText("匹配目标点...")
            matchedPairs = FeaturesDetect.match(currentPoints, totalOrderA, 0.55, 2.2)

            if len(currentPoints) < len(targetPoints):
                self.statusBar.showMessage(f"{len(targetPoints)}个目标点，但视场中仅识别到{len(currentPoints)}个点，仅对上述点进行就近匹配")
            else:
                self.statusBar.showMessage(f"{len(targetPoints)}个目标点已全部完成就近匹配")

            maxIterNum = self.maxIterNumInput.value()
            iterTarget = self.iterTargetInput.value() * 0.01

            self.secondStatusInfo.setText("计算路径帧...")

            self._frameGenerator = FrameGeneratorWorker(matchedPairs, self._framePipeSender)

            self._holoGenerator = HoloGeneratorWorker(
                self._framePipeReceiver,
                self._holoPipeSender,
                maxIterNum,
                iterTarget
            )

            self.progressBar.setRange(0,0)

            startThreads()

    def stopThreads(self):
        self._frameGenerator.terminate()
        self._holoGenerator.terminate()
        self._imgPlayTimer.stop()
        self._framePipeReceiver = None
        self._framePipeSender = None
        self._holoPipeReceiver = None
        self._holoPipeSender = None
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(100)
        self.secondStatusInfo.setText(f"计算已完成")
        logHandler.warning("Calculation terminated")
        self.autoCalcBtn.setText("从相机捕获")
        self.autoCalcBtn.clicked.disconnect()
        self.autoCalcBtn.clicked.connect(self.autoCalcHoloImg)


class SecondMonitorWindow(QMainWindow):
    """
    [副屏窗口类] 用于显示全息图
    """

    def __init__(self):
        super().__init__()

        self._selMonIndex = None
        self._initUI()

    def _initUI(self):
        self.setWindowTitle("Second Window")
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
        logHandler.debug(f"Image has been received from the main window.")
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
                    f'未检测到名为 "JDC EDK" 的LCOS硬件。\n'
                    f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                )
                logHandler.error(
                    f"No LCOS named 'JDC EDK' detected. "
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

                QMessageBox.warning(
                    self,
                    '警告',
                    f'程序被配置为[绕过LCOS设备检测]模式。\n'
                    f'当前监视器被设置为LCOS监视器，屏幕可能闪烁。该功能仅供开发时使用。'
                )
                logHandler.warning(f"Bypass LCOS detection mode, current monitor selected.")

            else:
                QMessageBox.critical(
                    self,
                    '错误',
                    f'未检测到名为 "JDC EDK" 的LCOS硬件。\n'
                    f'请重新连接SLM，确认电源已打开，或在显示设置中确认配置正确。'
                )
                logHandler.error(
                    f"No LCOS named 'JDC EDK' detected. "
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
