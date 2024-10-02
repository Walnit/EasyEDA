from PyQt6.QtCore import QSize, Qt, QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit, QMessageBox, QListWidget
import sys, time, os
import serial.tools.list_ports

class ScanWorker(QObject):
    finished = pyqtSignal()
    found = pyqtSignal(str)
    
    def run(self):
        for x in list(serial.tools.list_ports.comports()):
            self.found.emit(x.device)
        self.finished.emit()

class ScanWindow(QMainWindow):

    ports = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Device Scan")
        self.setMinimumSize(QSize(800, 450))

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        main_label = QLabel("Device Scan")
        main_label_font = main_label.font()
        main_label_font.setPointSize(24)
        main_label_font.setBold(True)
        main_label.setFont(main_label_font)
        layout.addWidget(main_label)

        self.label = QLabel("Scanning, please wait...")
        layout.addWidget(self.label)

        self.devices_list = QListWidget()
        self.devices_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.devices_list)

        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.done_btn_clicked)
        self.done_btn.setEnabled(False)
        layout.addWidget(self.done_btn)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.scan_devices()

    def done_btn_clicked(self):
        self.ports.emit([x.text() for x in list(self.devices_list.selectedItems())])
        self.close()

    def scan_devices(self):
        self.thread = QThread()
        self.worker = ScanWorker()
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.found.connect(
            lambda port: self.devices_list.addItem(port)
        )

        self.thread.start()
        self.thread.finished.connect(self.scan_finished)

    def scan_finished(self):
        self.label.setText("Please select the ports you wish to use, then press 'Done':")
        self.done_btn.setEnabled(True)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScanWindow()
    window.show()
    sys.exit(app.exec())
