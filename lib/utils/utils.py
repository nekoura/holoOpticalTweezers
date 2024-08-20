import os
import argparse
import logging
import colorlog
import math
import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QDir, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMessageBox
import queue


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
            prog='calcHoloGUI',
            description='SLM Holograph Generator',
            epilog=f'\tloglevel:\n'
                   f'\t10\t DEBUG\n'
                   f'\t20\t INFO\n'
                   f'\t30\t WARNING (default for console)\n'
                   f'\t40\t ERROR\n'
                   f'\t50\t FATAL',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            '-c', '--cloglvl', default=30, type=int,
            choices=(10, 20, 30, 40, 50),
            required=False, help='Set console log level'
        )

        parser.add_argument(
            '-bs', '--bypass-LCOS-detection', default=False, action='store_true',
            required=False, help='Bypass LCOS detection (set current monitor as LCOS, for development only)'
        )

        args = parser.parse_args()
        return args

    @staticmethod
    def folderPathCheck(path: str):
        """
        [工具类] 确认目标路径已被创建

        :param path: 输入路径
        :return: 输出路径
        :rtype: str
        """

        filePath = Path(path)
        if not filePath.exists():
            if filePath.suffix:
                filePath.parent.mkdir(parents=True, exist_ok=True)
            else:
                filePath.mkdir(parents=True, exist_ok=True)

        return path

    @staticmethod
    def getLog(consoleLevel=logging.INFO):
        """
        [工具类] log组件

        :param int consoleLevel: 控制台日志级别
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
                    f"[%(lineno)d]: %(message)s",
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

        return logger

    @staticmethod
    def exceptHook(cls, exception, traceback):
        print(f'\n')
        sys.__excepthook__(cls, exception, traceback)

    @staticmethod
    def exceptionHandler(errtype, value, traceback):
        # 捕获异步异常并显示错误消息
        QMessageBox.critical(None, 'Error', f"{errtype}\n{value}\n{traceback}")

    @staticmethod
    def calcEucDist(p1, p2) -> float:
        """
        [工具类] 计算两点之间的欧几里得距离

        :param p1: 点 1 坐标
        :param p2: 点 2 坐标
        :return: 距离
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def calcFileNum(path) -> int:
        """
        [工具类] 计算目录内文件数量

        :param path: 目录
        :return: 文件数量
        """
        return len(os.listdir(path))

    @staticmethod
    def timer(func):
        """
        [工具类] 修饰器：函数计时
        """
        def func_wrapper(*args, **kwargs):
            from time import time
            time_start = time()
            result = func(*args, **kwargs)
            time_end = time()
            time_spend = time_end - time_start
            print('%s cost time: %.3f s' % (func.__name__, time_spend))
            return result

        return func_wrapper


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

