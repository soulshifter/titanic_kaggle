import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("/home/vic/Desktop/Kaggle/Hubble_Wages/wages_hours.csv",sep="\t")
"""print(df1.head())"""

df2 = df1[["AGE", "RATE"]]
"""print(df2.head())"""

df2 = df2.sort_values(["AGE"], ascending=[True])
df2.set_index("AGE",inplace=True)

"""print(df2.head())"""
df2.plot()
plt.show()