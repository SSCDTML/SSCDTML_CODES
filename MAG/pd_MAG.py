from PyQt5.QtCore import pyqtSignal, QObject

class pd_MAG(QObject):
    valor_PD_MAG = pyqtSignal(float)

