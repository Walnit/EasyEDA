import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
# Black magic to calculate number of seconds
secs = pd.to_numeric(df[df.EDA.isna()].Time.str.split().str.get(3)).sum()
print(secs, "seconds in total")
for col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()
df.index = df.index + 1
new_row = pd.DataFrame({'Time':f"BEGIN {sys.argv[1]} FOR {secs} SECONDS", 'EDA':0, 'Heart Rate':0, "Temperature": 0}, index=[0])
df = pd.concat([new_row, df]).reset_index(drop = True)
print(df)
df.to_csv(sys.argv[1], index=False)
