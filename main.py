import sys
import csv
import time
import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

if len(sys.argv) != 2:
    print("Usage: main.py <port, e.g. /dev/rfcomm0 (Linux), COM3 (Windows)>")
    print("Try 'python -m serial.tools.list_ports' to list ports")
    sys.exit(1)

port = sys.argv[1]
app = pg.mkQApp("EasyEDA")
win = pg.GraphicsLayoutWidget(show=True, title="Sensor Signals")
win.resize(1000,600)
pg.setConfigOptions(antialias=True)

plot = win.addPlot()
eda_curve = plot.plot(pen=(0, 0, 255), name="EDA")
hr_curve = plot.plot(pen=(0, 255, 0), name="Heart Rate")
temp_curve = plot.plot(pen=(255, 0, 0), name="Temperature")

eda_history = np.array([], dtype=np.int32)
hr_history = np.array([], dtype=np.int32)
temp_history = np.array([], dtype=np.int32)
time_history = np.array([], dtype=np.int64)

def update(ser, csvwriter):
    data = ser.readline().strip()
    try:
        eda, hr, temp = data.split()
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
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: update(ser, csvwriter))
        timer.start(1)
        pg.exec()

