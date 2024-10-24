from PyQt6.QtCore import QSize, Qt, QObject, pyqtSignal, QThread, QMutex, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit, QMessageBox, QGridLayout, QComboBox
from qt_helpers import *
from qt_windows.SetupWindow import SetupWindow
import sys, traceback

app = QApplication(sys.argv)

window = SetupWindow()
window.show()

app.exec()

