"""
### 2 ###
Denne animation demonstrerer at mit måle-udstyr ikke var kalibret. Der forventes en vinkelhastighed på nul, når pendulet står stille. Det står stille i slutningen (~170-190s) og antiderivativen svarende til vinklen burde altså være en vandret linje, men der ses en tydelig hældning på linjen, når der zoomes ind. Hældningen på denne linje svarer til det vinklehastigheden er offsat med.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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


def antiderivative(xs, ys):
    h = np.diff(xs)
    a1 = ys[:-1]
    a2 = ys[1:]
    trapez_sum = 0.5 * (a1 + a2) * h
    integrals = np.concatenate(([0], np.cumsum(trapez_sum)))
    return integrals  # dvs. det ubestemte integral for y(x) gennem (xs[0],0)


antiderivative1 = antiderivative(ts1, omegas1)

mask_line = (ts1 > 170) & (ts1 < 190)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


fig, ax = plt.subplots()
plt.plot(ts1, antiderivative1, color='C0')
plt.plot(ts1[mask_line], antiderivative1[mask_line], color="red")
plt.title("Antiderivativen svarende til vinklen")
plt.show()

