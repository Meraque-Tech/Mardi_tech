import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("lio_global_pose_latlon.csv")

# Basic check
print(df.head())

# Plot GPS path
plt.figure()
plt.plot(df["longitude"], df["latitude"], marker='o')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GPS Path")
plt.axis("equal")   # Important for correct map aspect
plt.grid(True)

plt.show()
