import serial
import serial.tools.list_ports
from PyQt6.QtCore import QObject, pyqtSignal


class LaserMiddleWare(QObject):
    """
    [Laser类] 激光器控制中间件

    :var port: 当前设备端口
    :var portList: 设备端口列表
    :var device: 当前设备实例
    """

    def __init__(self):
        super().__init__()

        self.port = None
        self.portList = None

        self.device = None
        self.isEmitting = False

    def listComPorts(self):
        """
        [Laser类] 获取设备实例

        :raise IndexError: 无可用设备
        """
        self.portList = list(serial.tools.list_ports.comports())
        if len(self.portList) <= 0:
            raise IndexError

    def openComPort(self):
        """
        [Laser类] 打开设备端口

        :raise IOError: 打开设备失败
        """
        self.device = serial.Serial(self.port, 9600)
        if not self.device.isOpen():
            raise IOError

    def closeComPort(self):
        """
        [Laser类] 关闭设备端口
        """
        if self.device and self.device.isOpen():
            self.device.close()

    def setConfig(self, cmd):
        """
        [Laser类] 发送控制指令

        :param string cmd: 指令
        """
        data = [0x55, 0xAA]
        if cmd == 'ON':
            data.extend([0x03, 0x01, 0x04])
            self.isEmitting = True
        elif cmd == 'OFF':
            data.extend([0x03, 0x00, 0x03])
            self.isEmitting = False
        else:
            hexDigit = format(cmd, '04x')
            hDigit = int(hexDigit[:2], 16)
            lDigit = int(hexDigit[2:], 16)
            chkSum = 0x05 + 0x01 + hDigit + lDigit
            data.extend([0x05, 0x01, hDigit, lDigit, chkSum])

        self.device.write(data)
