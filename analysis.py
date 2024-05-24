import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv").astype("int")
df.iloc[:, 1:].plot()
#plt.savefig("figure.png")
plt.show()
