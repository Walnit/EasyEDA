import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
df.iloc[1:,1] = pd.to_numeric(df.iloc[1:,1],errors="coerce")
df.iloc[1:,2] = pd.to_numeric(df.iloc[1:,2],errors="coerce")
df.iloc[1:,3] = pd.to_numeric(df.iloc[1:,3],errors="coerce")
ls = [i for i, x in enumerate(df.set_index('Time').index.str.contains("BEGIN")) if x]
print(df.set_index('Time').index)
prev = 0
dfls = []
for index, i in enumerate(ls):
    print(i, index)
    df.iloc[prev:i,:].to_csv(f"file_{index}.csv", index=False)
    prev = i
df.iloc[:, 1:].plot()
#plt.savefig("figure.png")
plt.show()

