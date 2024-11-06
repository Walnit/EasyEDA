import csv, sys

time = 0

with open("out.csv", "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["Time", "EDA", "PPG", "Temp"])
    for i in range(1, len(sys.argv)):
        with open(sys.argv[i]) as data_file:
            for raw_data in data_file:
                data = raw_data.split()
                if len(data) == 4:
                    if data[0].isdigit() and data[1].isdigit() and data[2].isdigit() and data[3].isdigit(): 
                        time += int(data[0])
                        csvwriter.writerow([time, data[2], data[1], data[3]]) # Flip orders for EDA and PPG
                    else:
                        print("Not processing", data)
                else:
                    print("Not processing", data)
