import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
df.iloc[1:,1] = pd.to_numeric(df.iloc[1:,1],errors="coerce")
df.iloc[1:,2] = pd.to_numeric(df.iloc[1:,2],errors="coerce")
df.iloc[1:,3] = pd.to_numeric(df.iloc[1:,3],errors="coerce")
df.dropna(inplace=True)
ls = [i for i, x in enumerate(df.set_index('Time').index == "BEGIN baseline FOR 120 SECONDS") if x]
prev = 0
dfls = []
for i in ls:
    dfls.append(df.iloc[prev:i,:])
    prev = i
df.iloc[:, 1:].plot()
#plt.savefig("figure.png")
plt.show()

