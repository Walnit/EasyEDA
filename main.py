import sys
import csv
import time
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if len(sys.argv) != 2:
    print("Usage: main.py <port, e.g. /dev/rfcomm0 (Linux), COM3 (Windows)>")
    print("Try 'python -m serial.tools.list_ports' to list ports")
    sys.exit(1)

port = sys.argv[1]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

eda_history = []
hr_history = []
temp_history = []
time_history = []
sampling = 0

def get_data(ser):
    while True:
        data = ser.readline().strip()
        try:
            eda, hr, temp = data.split()
            return int(eda), int(hr), int(temp)
        except:
            print("Malformed input, skipping")

def plot_data(_, ax, ser, csvwriter):
    # Get data, write to file
    eda, hr, temp = get_data(ser)
    now = time.time_ns() // 1000 # microseconds since epoch
    csvwriter.writerow([now, eda, hr, temp])
    global sampling
    if sampling % 10 == 9: # only graph every 10 frames for performance
        # Pop first element if buffer too large
        if len(eda_history) > 200:
            eda_history.pop(0)
            hr_history.pop(0)
            temp_history.pop(0)
            time_history.pop(0)

        # Write to buffer
        eda_history.append(eda)
        hr_history.append(hr)
        temp_history.append(temp)
        time_history.append(now)

        # Plot buffer
        ax.clear()
        ax.plot(time_history,eda_history, color="blue")
        ax.plot(time_history, hr_history, color="green")
        ax.plot(time_history, temp_history, color="red")

    sampling += 1


with serial.Serial(port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
    print(ser)
    with open("data.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["Time", "EDA", "Heart Rate", "Temperature"])
        an = animation.FuncAnimation(fig, plot_data, fargs=(ax, ser, csvwriter), interval=0, blit=True)
        plt.show()

