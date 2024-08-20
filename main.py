import sys
import csv
import time
import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from scheduler import parse_schedule

if len(sys.argv) != 2:
    print("Usage: main.py <port, e.g. /dev/rfcomm0 (Linux), COM3 (Windows)>")
    print("Try 'python -m serial.tools.list_ports' to list ports")
    sys.exit(1)

port = sys.argv[1]
app = pg.mkQApp("EasyEDA")
win = QtWidgets.QMainWindow()
win.resize(1000,600)
plot_widget = pg.GraphicsLayoutWidget(show=True, title="Sensor Signals")

cw = QtWidgets.QWidget()
win.setCentralWidget(cw)
l = QtWidgets.QHBoxLayout()
cw.setLayout(l)

menu = QtWidgets.QVBoxLayout()
curr_text = QtWidgets.QLabel("Paused, press next")
next_btn = QtWidgets.QPushButton("Next")
menu.addWidget(curr_text)
menu.addWidget(next_btn)

l.addWidget(plot_widget)
l.addLayout(menu)

pg.setConfigOptions(antialias=True)
win.show()

plot = plot_widget.addPlot()
eda_curve = plot.plot(pen=(0, 0, 255), name="EDA")
hr_curve = plot.plot(pen=(0, 255, 0), name="Heart Rate")
temp_curve = plot.plot(pen=(255, 0, 0), name="Temperature")
plot.enableAutoRange('y', True)


eda_history = np.array([], dtype=np.int32)
hr_history = np.array([], dtype=np.int32)
temp_history = np.array([], dtype=np.int32)
time_history = np.array([], dtype=np.int64)

# Scheduling 
schedule = parse_schedule("schedule.txt")
schedule_index = 0
previous_time = 0
print(schedule)

def next():
    global previous_time, curr_text, schedule_index, schedule
    schedule_index += 1
    if schedule_index >= len(schedule):
        # Quit App
        pg.exit()

    if schedule[schedule_index][0] != 'pause':
        previous_time = time.time()
        csvwriter.writerow([f"BEGIN {schedule[schedule_index][1]} FOR {schedule[schedule_index][0]} SECONDS", "", "", ""])
        curr_text.setText(f"{schedule[schedule_index][1]} for {schedule[schedule_index][0]}s")
    else:
        previous_time = 0
        curr_text.setText("Paused, press next")

next_btn.clicked.connect(next)

def update(ser, csvwriter):
    data = ser.readline().strip()
    global schedule, schedule_index, previous_time, curr_text
    if schedule[schedule_index][0] != 'pause':
        try:
            # Schedule stuff
            if time.time() - previous_time > int(schedule[schedule_index][0]):
                next()

            # Data stuff
            raw_values = data.split()
            if len(raw_values) == 3:
                eda, hr, temp = raw_values
            elif len(raw_values) == 4:
                _, eda, hr, temp = raw_values
            else:
                print("Uh oh debugging time", raw_values)
            eda, hr, temp = int(eda), int(hr), int(temp)
            now = time.time_ns() // 1000 # microseconds since epoch
            csvwriter.writerow([now, eda, hr, temp])
            global eda_history, hr_history, temp_history, time_history, eda_curve, hr_curve, temp_curve
            # Pop first element if buffer too large
            if eda_history.size > 200:
                plot.enableAutoRange('xy', False)
                eda_history = np.delete(eda_history, 0)
                hr_history = np.delete(hr_history, 0)
                temp_history = np.delete(temp_history, 0)
                time_history = np.delete(time_history, 0)

            # Write to buffer
            eda_history = np.append(eda_history, eda)
            hr_history = np.append(hr_history, hr)
            temp_history = np.append(temp_history, temp)
            time_history = np.append(time_history, now)

            eda_curve.setData(eda_history)
            hr_curve.setData(hr_history)
            temp_curve.setData(temp_history)

        except ValueError:
            print("Malformed input, skipping")


with serial.Serial(port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
    print(ser)
    with open("data.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["Time", "EDA", "Heart Rate", "Temperature"])
        if schedule[schedule_index][0] != 'pause':
            previous_time = time.time()
            curr_text.setText(f"{schedule[schedule_index][1]} for {schedule[schedule_index][0]}s")
            csvwriter.writerow([f"BEGIN {schedule[schedule_index][1]} FOR {schedule[schedule_index][0]} SECONDS", "", "", ""])
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: update(ser, csvwriter))
        timer.start(1)
        pg.exec()


