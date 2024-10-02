from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel

def get_bold_label(text):
    bold_label = QLabel(text)
    bold_label_font = bold_label.font()
    bold_label_font.setBold(True)
    bold_label.setFont(bold_label_font)
    return bold_label