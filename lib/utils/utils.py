import os
import sys
import time
import argparse
import logging
import colorlog
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class Utils:
    """
    [工具类] 处理文件和基本图像操作
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def getCmdOpt():
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
            '-d', '--autodetect', default=False, action='store_true',
            required=False, help='Auto detect LCOS monitor'
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
        pathPrefix = os.path.split(path)[0]
        if os.path.exists(pathPrefix):
            pass
        else:
            os.makedirs(pathPrefix)

        return path

    def getLog(self, consoleLevel=logging.WARNING, fileLevel=logging.DEBUG, writeLogFile=False):
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
                filename=self.folderPathCheck(f"../log/log_{time.strftime('%Y%m%d%H%M%S')}.txt"),
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
    def loadImg(string):
        """
        [工具类] 从文件加载图像

        :param str string: 输入图像路径
        """
        # img = cv2.imread(string)
        img = cv2.imdecode(np.fromfile(string, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # if img.shape[2] > 1:
            #     imgBW = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, None)
            #     # logHandler.debug('RGB image')
            #     return imgBW
            # else:
            #     # logHandler.debug('Gray image')
            #     return img
            return img
        else:
            raise IOError

    @staticmethod
    def cvImg2QPixmap(obj, inputImg):
        """
        [工具类] 加载cv2图像到Qt界面

        :param QObject obj: 目标Qt组件
        :param inputImg: 输入图像
        """
        if inputImg is not None:
            if len(inputImg.shape) == 2:
                # 灰度图是单通道，所以需要用Format_Indexed8
                rows, columns = inputImg.shape
                bytesPerLine = columns
                img = QImage(
                    inputImg.data, columns, rows, bytesPerLine, QImage.Format.Format_Indexed8
                )
            else:
                rows, columns, channels = inputImg.shape
                bytesPerLine = channels * columns
                img = QImage(
                    inputImg.data, columns, rows, bytesPerLine, QImage.Format.Format_RGB888
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