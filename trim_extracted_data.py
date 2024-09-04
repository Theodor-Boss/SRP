"""
### 3 ###
Denne fil trimmer dataene, da rystelserne i starten og slutningen ikke skal bruges.
"""
import numpy as np
import matplotlib.pyplot as plt



extracted_data1 = "extracted_data1.npz"
extracted_data2 = "extracted_data2.npz"
extracted_data3 = "extracted_data3.npz"
extracted_data4 = "extracted_data4.npz"
extracted_data5 = "extracted_data5.npz"

with np.load(extracted_data1) as data:
    ts1 = data["ts"]
    omegas1 = data["omegas"]

with np.load(extracted_data2) as data:
    ts2 = data["ts"]
    omegas2 = data["omegas"]

with np.load(extracted_data3) as data:
    ts3 = data["ts"]
    omegas3 = data["omegas"]

with np.load(extracted_data4) as data:
    ts4 = data["ts"]
    omegas4 = data["omegas"]

with np.load(extracted_data5) as data:
    ts5 = data["ts"]
    omegas5 = data["omegas"]

