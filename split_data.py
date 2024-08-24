import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Please enter the name of the file you wish to split!")
else:
    df = pd.read_csv(sys.argv[1])
    df.iloc[1:,1] = pd.to_numeric(df.iloc[1:,1],errors="coerce")
    df.iloc[1:,2] = pd.to_numeric(df.iloc[1:,2],errors="coerce")
    df.iloc[1:,3] = pd.to_numeric(df.iloc[1:,3],errors="coerce")
    ls = [i for i, x in enumerate(df.set_index('Time').index.str.contains("BEGIN")) if x]
    prev = 0
    dfls = []
    for index, i in enumerate(ls):
        if i == 0: continue
        section = df.iloc[prev,0].split()[1]
        print(f"{index}: {section}")
        df.iloc[prev:i,:].to_csv(f"File{index}_{section}.csv", index=False)
        prev = i
    print("Split complete!")
