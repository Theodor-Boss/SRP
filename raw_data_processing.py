"""
### 1 ###
Denne fil opretter for hver csv-fil en .npz-fil, der kun indeholder tids-værdierne og vinkelhastighed-værdierne.
"""
import numpy as np


def extract_data_from_csv(file_path, read_from=4):
    with open(file_path, "r") as csvfile:
        lines = csvfile.readlines()

    t0 = float(lines[read_from].split(",")[0])  # Starttidspunkt

    ts = np.empty(len(lines[read_from:]))
    omegas = np.empty_like(ts)

    for i, line in enumerate(lines[read_from:]):
        time = float(line.split(",")[0]) - t0  # Tid er i 0. kolonne
        omega = line.split(",")[6]  # Vinkelhastighed er i 6. kolonne
        ts[i] = time
        omegas[i] = omega

    return ts, omegas


file_path1 = "nyserie1.csv"
ts1, omegas1 = extract_data_from_csv(file_path1)

file_path2 = "nyserie2.csv"
ts2, omegas2 = extract_data_from_csv(file_path2)

file_path3 = "nyserie3.csv"
ts3, omegas3 = extract_data_from_csv(file_path3)

file_path4 = "nyserie4.csv"
ts4, omegas4 = extract_data_from_csv(file_path4)

file_path5 = "nyserie5.csv"
ts5, omegas5 = extract_data_from_csv(file_path5)

extracted_data1 = "extracted_data1.npz"
extracted_data2 = "extracted_data2.npz"
extracted_data3 = "extracted_data3.npz"
extracted_data4 = "extracted_data4.npz"
extracted_data5 = "extracted_data5.npz"

np.savez(extracted_data1, ts=ts1, omegas=omegas1)
np.savez(extracted_data2, ts=ts2, omegas=omegas2)
np.savez(extracted_data3, ts=ts3, omegas=omegas3)
np.savez(extracted_data4, ts=ts4, omegas=omegas4)
np.savez(extracted_data5, ts=ts5, omegas=omegas5)
