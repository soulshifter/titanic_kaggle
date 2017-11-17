import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("/home/vic/Desktop/Kaggle/Hubble_Wages/hubble_data.csv")
"""print(df1.head())"""

df1.set_index("distance",inplace=True)

"""print(df1.head())"""

df1.plot()
plt.show()