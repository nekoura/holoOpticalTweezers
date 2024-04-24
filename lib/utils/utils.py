import sys
import time
import argparse
import logging
import colorlog
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMessageBox


class Utils:
    """
    [工具类] 处理文件和基本图像操作
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def getCmdOpt() -> object:
        """
        [工具类] 处理命令行参数

        :return: (终端日志级别, 文件日志级别, 日志是否写入文件)
        :rtype: tuple
        """
        parser = argparse.ArgumentParser(
            prog='calcHoloGUI',  # 程序名
            description='SLM Holograph Generator',  # 描述
            epilog=f'\tloglevel:\n'
                   f'\t10\t DEBUG\n'
                   f'\t20\t INFO\n'
                   f'\t30\t WARNING (default for console)\n'
                   f'\t40\t ERROR\n'
                   f'\t50\t FATAL',  # 说明信息
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            '-c', '--cloglvl', default=30, type=int,
            choices=(10, 20, 30, 40, 50),
            required=False, help='Set console log level'
        )
        parser.add_argument(
            '-f', '--floglvl', default=-1, type=int,
            choices=(10, 20, 30, 40, 50),
            required=False, help='Set logfile log level and enable logfile'
        )
        parser.add_argument(
            '-d', '--auto-detect', default=False, action='store_true',
            required=False, help='Auto detect LCOS devices'
        )

        parser.add_argument(
            '-bs', '--bypass-LCOS-detection', default=False, action='store_true',
            required=False, help='Bypass LCOS detection (set current monitor as LCOS, for development only)'
        )

        parser.add_argument(
            '-bl', '--bypass-laser-detection', default=False, action='store_true',
            required=False, help='Bypass Laser detection (if laser control is unnecessary)'
        )

        args = parser.parse_args()
        return args

    @staticmethod
    def folderPathCheck(path):
        """
        [工具类] 确认目标路径已被创建

        :param str path: 输入路径
        :return: 输出路径
        :rtype: str
        """
        parent = Path(path).parent
        if Path(parent).exists:
            pass
        else:
            Path.mkdir(parent)

        return path

    @staticmethod
    def getLog(consoleLevel=logging.WARNING, fileLevel=logging.DEBUG, writeLogFile=False):
        """
        [工具类] log组件

        :param bool writeLogFile: 是否写入log文件
        :param int consoleLevel: 控制台日志级别
        :param int fileLevel: 文件日志级别
        :return: 日志句柄
        """
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # 输出到控制台
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(consoleLevel)
        consoleHandler.setFormatter(
            colorlog.ColoredFormatter(
                fmt=f"%(log_color)s[%(levelname)s] [%(asctime)s] "
                    f"%(module)s->%(funcName)s (line %(lineno)d): %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_white,bg_red',
                }
            )
        )
        logger.addHandler(consoleHandler)

        # log打印到文件
        if writeLogFile is True:
            logFileHandler = logging.FileHandler(
                filename=Utils.folderPathCheck(f"../log/log_{time.strftime('%Y%m%d%H%M%S')}.txt"),
                encoding='utf8'
            )
            logFileHandler.setLevel(fileLevel)
            logFileHandler.setFormatter(
                logging.Formatter(
                    fmt=f"[%(levelname)s] [%(asctime)s.%(msecs)03d] "
                        f"%(module)s->%(funcName)s (line %(lineno)d): %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            logger.addHandler(logFileHandler)
            logger.info(f"log will print to file '../log/log_{time.strftime('%Y%m%d%H%M%S')}.txt'")

        return logger

    @staticmethod
    def exceptHook(cls, exception, traceback):
        print(f'\n')
        sys.__excepthook__(cls, exception, traceback)

    @staticmethod
    def exceptionHandler(errtype, value, traceback):
        # 捕获异步异常并显示错误消息
        QMessageBox.critical(None, 'Error', f"{errtype}\n{value}\n{traceback}")


class ImgProcess:
    @staticmethod
    def loadImg(string):
        """
        [图像处理] 从文件加载图像

        :param str string: 输入图像路径
        """
        img = cv2.imdecode(np.fromfile(string, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        else:
            raise IOError

    @staticmethod
    def cvImg2QPixmap(obj, inputImg):
        """
        [图像处理] 加载cv2图像到Qt界面

        :param QObject obj: 目标Qt组件
        :param inputImg: 输入图像
        """
        if inputImg is not None:
            if len(inputImg.shape) == 2:
                # 灰度图是单通道，所以需要用Format_Indexed8
                rows, columns = inputImg.shape
                bytesPerLine = columns
                img = QImage(
                    inputImg.copy(), columns, rows, bytesPerLine, QImage.Format.Format_Indexed8
                )
            else:
                rows, columns, channels = inputImg.shape
                bytesPerLine = channels * columns
                img = QImage(
                    inputImg.copy(), columns, rows, bytesPerLine, QImage.Format.Format_RGB888
                ).rgbSwapped()

            pixmap = QPixmap(img)
        else:
            # 如果图片未加载，设置透明图为占位符
            pixmap = QPixmap(1, 1)
            pixmap.fill(Qt.GlobalColor.transparent)

        scaledPixmap = pixmap.scaled(
            obj.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        obj.setPixmap(scaledPixmap)

    def createGaussianBeamMask(self, image, beamWidth: int):
        """
        [图像处理] 生成以指定位置为中心的高斯光束的遮罩
        :param image: 输入图像
        :param beamWidth: 光斑直径
        """
        circles = self.detectCircles(image)
        size = image.shape[:2]
        combinedMask = np.zeros(size, dtype=np.float32)

        X, Y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

        for x, y, r in circles:
            rSquared = ((X - x) ** 2 + (Y - y) ** 2)
            phase = np.exp(-rSquared / (2 * beamWidth ** 2))
            combinedMask += phase

        return combinedMask

    @staticmethod
    def detectCircles(image):
        """
        [图像处理] 使用霍夫变换在图像中检测圆形
        """
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=20, maxRadius=40
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        else:
            return []
