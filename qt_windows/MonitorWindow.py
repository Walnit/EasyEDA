from PyQt6.QtCore import QSize, Qt, QObject, pyqtSignal, QThread, QMutex, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit, QMessageBox, QGridLayout, QComboBox
import sys, serial, traceback, time
import pyqtgraph as pg
import numpy as np

class SerialWorker(QObject):
    finished = pyqtSignal()
    initialized = pyqtSignal()
    report = pyqtSignal(bytes)
    failed = pyqtSignal(str, QObject)

    def __init__(self, id, tty, step_mutex, step_ptr):
        super().__init__()
        self.id = id
        self.tty = tty
        self.step_mutex = step_mutex
        self.step_ptr = step_ptr

    def run(self):
        self.step_mutex.lock()
        step = self.step_ptr[0]
        self.step_mutex.unlock()

        try:
            with serial.Serial(self.tty, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
                while step != -1:
                    with open(f"{step}_id-{self.id}.log", "wb") as f:
                        ser.readline()
                        self.initialized.emit()
                        print(f"ID {self.id} initialized for step {step}!")
                        while True:
                            for i in range(100):
                                data = ser.readline()
                                f.write(data)
                                if i % 20 == 19:
                                    self.report.emit(data)
                            f.flush() # Flush every 100 samples (~2 seconds)
                            
                            
                            self.step_mutex.lock()
                            if step != self.step_ptr[0]:
                                step = self.step_ptr[0]
                                self.step_mutex.unlock()
                                break
                            self.step_mutex.unlock()
            print(f"ID {self.id} exiting!")
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(traceback.format_exc()), self)



class MonitorWindow(QMainWindow):
    def __init__(self, participants_dict, schedule_list):
        super().__init__()

        self.participants_id = list(participants_dict.keys())
        self.participants_tty = [participants_dict[id] for id in self.participants_id]
        self.schedule_list = schedule_list

        self.setWindowTitle("EasyEDA - Monitoring")
        self.setMinimumSize(QSize(1000, 600))

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        main_label = QLabel("Live Monitor")
        main_label_font = main_label.font()
        main_label_font.setPointSize(24)
        main_label_font.setBold(True)
        main_label.setFont(main_label_font)
        layout.addWidget(main_label)

        # INSERT GRAPH HERE
        plot_widget = pg.GraphicsLayoutWidget(show=True, title="Sensor Signals")
        pg.setConfigOptions(antialias=True)

        self.plot = plot_widget.addPlot()
        self.eda_curve = self.plot.plot(pen=(0, 0, 255), name="EDA")
        self.hr_curve = self.plot.plot(pen=(0, 255, 0), name="Heart Rate")
        self.temp_curve = self.plot.plot(pen=(255, 0, 0), name="Temperature")
        self.plot.enableAutoRange('y', True)

        self.eda_history = np.array([], dtype=np.int32)
        self.hr_history = np.array([], dtype=np.int32)
        self.temp_history = np.array([], dtype=np.int32)

        layout.addWidget(plot_widget)

        self.status_text = QLabel("Status: Initializing")
        layout.addWidget(self.status_text)

        grid = QGridLayout()

        id_row = QHBoxLayout()
        id_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        id_row.addWidget(QLabel("ID: "))
        id_dropdown = QComboBox()
        id_dropdown.addItems(self.participants_id)
        id_dropdown.currentIndexChanged.connect(self.id_dropdown_changed)

        id_row.addWidget(id_dropdown)
        id_row.addStretch()
        grid.addLayout(id_row, 0, 0)

        control_row = QHBoxLayout()
        control_row.setAlignment(Qt.AlignmentFlag.AlignRight)

        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_btn_clicked)
        control_row.addWidget(next_btn)

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.back_btn_clicked)
        control_row.addWidget(back_btn)

        grid.addLayout(control_row, 0, 1)

        layout.addLayout(grid)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.step = [0]
        self.devices_ready = 0
        self.failed_workers = 0
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.update_status())
        self.timer.start(1000)
        self.countdown = 0

        self.run_monitoring()
        self.id_dropdown_changed(0)

    def plot_data(self, data_bytes):
        data = data_bytes.strip()
        raw_values = data.split()
        if len(raw_values) == 3:
            hr, eda, temp = raw_values
        elif len(raw_values) == 4:
            _, hr, eda, temp = raw_values
        else: 
            return

        try:
            eda, hr, temp = int(eda), int(hr), int(temp)

            self.eda_history = np.append(self.eda_history, eda)
            self.hr_history = np.append(self.hr_history, hr)
            self.temp_history = np.append(self.temp_history, temp)

            # Pop first element if buffer too large
            if self.eda_history.size > 200:
                self.plot.enableAutoRange('xy', False)
                self.eda_history = np.delete(self.eda_history, 0)
                self.hr_history = np.delete(self.hr_history, 0)
                self.temp_history = np.delete(self.temp_history, 0)

            self.eda_curve.setData(self.eda_history)
            self.hr_curve.setData(self.hr_history)
            self.temp_curve.setData(self.temp_history)
        except ValueError as e:
            print("Invalid Input detected!")

    def run_monitoring(self):
        self.mutex = QMutex()

        self.workers = [SerialWorker(id, tty, self.mutex, self.step) for id, tty in zip(self.participants_id, self.participants_tty)]
        self.threads = []

        for worker in self.workers:
            thread = QThread()
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.initialized.connect(self.worker_ready)
            worker.finished.connect(self.experiment_completed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            worker.failed.connect(self.worker_failed)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)


            thread.start()
            self.threads.append(thread)

    def worker_failed(self, msg, worker):
        print(msg)
        print(worker)
        self.workers.remove(worker)
        self.devices_ready -= 1
        self.failed_workers += 1
        print(self.workers)


    def worker_ready(self):
        self.devices_ready += 1
        if self.devices_ready == len(self.participants_id):
            self.countdown = int(self.schedule_list[self.step[0]][0])
            self.update_status()
        else:
            self.status_text.setText(f"Status: {self.devices_ready}/{len(self.participants_id)} Ready")


    def experiment_completed(self):
        self.devices_ready += 1
        if self.devices_ready == len(self.participants_id):
            self.status_text.setText(f"Experiment Complete")
        else:
            self.status_text.setText(f"Status: {self.devices_ready}/{len(self.participants_id)} Ready")

    def update_status(self):
        self.mutex.lock()
        if self.devices_ready == len(self.participants_id):
            if self.countdown > 0:
                self.status_text.setText(f"Status: {self.schedule_list[self.step[0]][1]} {self.countdown}s remaining")
                self.countdown -= 1
            else:
                if self.step[0] == len(self.schedule_list) - 1:
                    self.status_text.setText(f"Status: Proceed to End")
                else:
                    self.status_text.setText(f"Status: Proceed to {self.schedule_list[self.step[0]+1][1]}")
        self.mutex.unlock()


    def id_dropdown_changed(self, index):
        for worker in self.workers:
            try:
                worker.report.disconnect()
            except TypeError: pass            
        self.workers[index].report.connect(self.plot_data)

        self.plot.enableAutoRange('y', True)
        self.eda_history = np.array([], dtype=np.int32)
        self.hr_history = np.array([], dtype=np.int32)
        self.temp_history = np.array([], dtype=np.int32)

    def next_btn_clicked(self):
        self.mutex.lock()
        if self.step[0] == len(self.schedule_list) - 1:
            self.step[0] = -1
            self.timer.stop()
        elif self.step[0] >= len(self.schedule_list):
            pass
        else:
            self.step[0] += 1
        self.mutex.unlock()

        self.devices_ready = 0
        self.status_text.setText(f"Status: {self.devices_ready}/{len(self.participants_id)} Ready")
        

    def back_btn_clicked(self):
        if self.devices_ready < len(self.participants_id):
            self.devices_ready += 1
        self.mutex.lock()
        if self.step[0] == 0:
            self.mutex.unlock()
            return
        self.step[0] -= 1
        self.mutex.unlock()

        self.devices_ready = 0
        self.status_text.setText(f"Status: {self.devices_ready}/{len(self.participants_id)} Ready")
