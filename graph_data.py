import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
df.iloc[:, 1:].plot()
#plt.savefig("figure.png")
plt.show()

