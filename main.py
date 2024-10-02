from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit, QMessageBox
from qt_helpers import *
from qt_windows.ScanWindow import ScanWindow
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.participant_widgets = []

        self.setWindowTitle("EasyEDA")
        self.setMinimumSize(QSize(1000, 600))

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        main_label = QLabel("Experiment Configuration")
        main_label_font = main_label.font()
        main_label_font.setPointSize(24)
        main_label_font.setBold(True)
        main_label.setFont(main_label_font)
        layout.addWidget(main_label)

        layout.addWidget(get_bold_label("Setup"))

        devices_row = QHBoxLayout()
        devices_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        devices_row.addWidget(QLabel("Number of Devices:"))

        self.num_devices_spin = QSpinBox()
        self.num_devices_spin.setMinimum(1)
        self.num_devices_spin.valueChanged.connect(self.num_devices_changed)
        self.num_devices_spin.setValue(1)
        devices_row.addWidget(self.num_devices_spin)

        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self.scan_btn_clicked)
        devices_row.addWidget(scan_btn)

        devices_row.addStretch()
        layout.addLayout(devices_row)

        self.devices_vbox = QVBoxLayout()
        devices_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(self.devices_vbox)
        self.num_devices_changed(1)

        begin_btn = QPushButton("Begin")
        begin_btn.clicked.connect(self.begin_btn_clicked)

        layout.addStretch()
        layout.addWidget(begin_btn)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def scan_btn_clicked(self):
        self.scan_window = ScanWindow()
        self.scan_window.ports.connect(self.load_scanned_ports)
        self.scan_window.show()

    def load_scanned_ports(self, ports):
        if len(ports) > 0:
            self.num_devices_spin.setValue(len(ports))
            self.clear_layout(self.devices_vbox)
            self.participant_widgets = []
            for i in range(1, len(ports)+1):
            
                participant_group = QVBoxLayout()

                participant_group.addWidget(get_bold_label(f"Device {i}"))

                id_row = QHBoxLayout()
                id_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
                id_row.addWidget(QLabel("Participant ID:"))
                
                id_edit = QLineEdit()
                id_row.addWidget(id_edit)

                id_row.addStretch()
                participant_group.addLayout(id_row)

                ser_row = QHBoxLayout()
                ser_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
                ser_row.addWidget(QLabel("Serial Port:"))
                
                ser_edit = QLineEdit()
                ser_edit.setText(ports[i-1])
                ser_row.addWidget(ser_edit)

                ser_row.addStretch()
                participant_group.addLayout(ser_row)

                self.devices_vbox.addLayout(participant_group)
                
                self.participant_widgets.append((id_edit, ser_edit))


    def begin_btn_clicked(self):
        results = {}
        
        # Check for any blanks
        has_blanks = False
        for id_edit, ser_edit in self.participant_widgets:
            if id_edit.text().strip() == "" or ser_edit.text().strip() == "":
                has_blanks = True
                break

        if not has_blanks:
            for id_edit, ser_edit in self.participant_widgets:
                results[id_edit.text().strip()] = ser_edit.text().strip()
                print(results)
        else:
            QMessageBox.critical(self, "Error", "Please fill out all fields!")


    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())


    def num_devices_changed(self, value):
        # Modify values to preserve existing values
        if value < len(self.participant_widgets):
            for i in range(len(self.participant_widgets)-value):
                self.clear_layout(self.devices_vbox.takeAt(self.devices_vbox.count()-1))
                self.participant_widgets.pop()

        for i in range(len(self.participant_widgets)+1, value+1):
            participant_group = QVBoxLayout()

            participant_group.addWidget(get_bold_label(f"Device {i}"))

            id_row = QHBoxLayout()
            id_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            id_row.addWidget(QLabel("Participant ID:"))
            
            id_edit = QLineEdit()
            id_row.addWidget(id_edit)

            id_row.addStretch()
            participant_group.addLayout(id_row)

            ser_row = QHBoxLayout()
            ser_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            ser_row.addWidget(QLabel("Serial Port:"))
            
            ser_edit = QLineEdit()
            ser_row.addWidget(ser_edit)

            ser_row.addStretch()
            participant_group.addLayout(ser_row)

            self.devices_vbox.addLayout(participant_group)
            
            self.participant_widgets.append((id_edit, ser_edit))




app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

