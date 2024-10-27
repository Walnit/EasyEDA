from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit, QMessageBox, QFileDialog
from qt_helpers import *
from qt_windows.ScanWindow import ScanWindow
from qt_windows.MonitorWindow import MonitorWindow
import os, traceback, serial

class SetupWindow(QMainWindow):
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

        schedule_row = QHBoxLayout()
        schedule_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        schedule_row.addWidget(QLabel("Experiment Schedule:"))
        self.schedule_edit = QLineEdit()
        self.schedule_edit.setText("schedule.txt")
        schedule_row.addWidget(self.schedule_edit)

        schedule_btn = QPushButton("Browse Files...")
        schedule_btn.clicked.connect(self.schedule_btn_clicked)
        schedule_row.addWidget(schedule_btn)

        schedule_row.addStretch()
        layout.addLayout(schedule_row)

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

                print (f"Testing port {ports[i-1]}")
                try:
                    for retries in range(3):
                        with serial.Serial(ports[i-1], baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=10) as ser:
                            invalids = 0
                            for n in range(500): # Read 5s of data
                                data = ser.readline().strip()
                                raw_values = data.split()
                                if len(raw_values) == 3:
                                    hr, eda, temp = raw_values
                                elif len(raw_values) == 4:
                                    _, hr, eda, temp = raw_values
                                else: 
                                    print("Invalid data read:", raw_values)
                                    invalids += 1

                        if invalids > 20:
                            if retries < 2:
                                print("Too many invalid reads, trying again...")
                            else:
                                raise Exception("Too many invalid reads")
                        else:
                            break

                    print("Testing passed!")

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

                except Exception as e:
                    print("Oh no, error!")
                    print(str(traceback.format_exc()))
                    QMessageBox.warning(self, f"Error with {ports[i-1]}", str(traceback.format_exc()))
            
                

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
                
            filename = self.schedule_edit.text().strip()
            if (os.path.isfile(filename)):
                parsed = []
                with open(filename) as f:
                    lines = f.read().splitlines()
                    parsed = [ line.split() for line in lines ]
                window = MonitorWindow(results, parsed)
                self.hide()
                window.show()
            else:
                QMessageBox.critical(self, "Error", "Unable to access schedule file!")

            
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

    def schedule_btn_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)")
        if filename:
            self.schedule_edit.setText(filename)