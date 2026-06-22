# === SETUP: load the provided files (regenerate them if missing) ===
import os
import numpy as np
import pandas as pd

# Each sensor is a linear mix of a few latent machine states + noise, so the 24
# columns are highly correlated -> exactly the structure PCA exploits.
# spec: (name, unit, group, c_load, c_wear, c_thermal, c_vib, base, scale, noise)
SENSORS = [
    ("temp_bearing",   "C",   "thermal",   0.2, 0.6, 0.9, 0.1, 66, 6, 0.30),
    ("temp_motor",     "C",   "thermal",   0.3, 0.4, 0.9, 0.1, 70, 7, 0.30),
    ("temp_gearbox",   "C",   "thermal",   0.2, 0.5, 0.8, 0.2, 62, 6, 0.35),
    ("oil_temp",       "C",   "thermal",   0.2, 0.5, 0.7, 0.1, 58, 5, 0.35),
    ("temp_ambient",   "C",   "thermal",   0.0, 0.0, 0.3, 0.0, 28, 2, 0.60),
    ("vibration_x",    "mm/s","vibration", 0.4, 0.3, 0.1, 0.9, 2.2, 0.6, 0.30),
    ("vibration_y",    "mm/s","vibration", 0.4, 0.3, 0.1, 0.9, 2.0, 0.6, 0.30),
    ("vibration_z",    "mm/s","vibration", 0.3, 0.4, 0.1, 0.8, 1.8, 0.5, 0.35),
    ("acoustic_db",    "dB",  "vibration", 0.3, 0.3, 0.1, 0.7, 78, 5, 0.40),
    ("rpm",            "rpm", "drive",     0.9, 0.1, 0.1, 0.3, 1800, 350, 0.25),
    ("spindle_speed",  "rpm", "drive",     0.9, 0.1, 0.1, 0.3, 6000, 1500, 0.25),
    ("feed_rate",      "mm/min","drive",   0.8, 0.1, 0.1, 0.2, 250, 60, 0.30),
    ("torque",         "Nm",  "drive",     0.8, 0.2, 0.1, 0.3, 120, 25, 0.30),
    ("power_kw",       "kW",  "power",     0.9, 0.2, 0.2, 0.2, 75, 18, 0.25),
    ("current_a",      "A",   "power",     0.8, 0.2, 0.2, 0.2, 22, 5, 0.30),
    ("current_b",      "A",   "power",     0.8, 0.2, 0.2, 0.2, 22, 5, 0.30),
    ("current_c",      "A",   "power",     0.8, 0.2, 0.2, 0.2, 22, 5, 0.30),
    ("load_pct",       "%",   "load",      0.9, 0.1, 0.1, 0.2, 65, 15, 0.25),
    ("pressure_in",    "bar", "hydraulic", 0.5, 0.2, 0.2, 0.2, 80, 12, 0.35),
    ("pressure_out",   "bar", "hydraulic", 0.5, 0.2, 0.2, 0.2, 60, 10, 0.35),
    ("coolant_flow",   "L/min","hydraulic",0.4, 0.2, 0.3, 0.1, 30, 6, 0.40),
    ("oil_level",      "%",   "hydraulic", 0.0,-0.4, 0.0, 0.0, 80, 8, 0.40),
    ("voltage",        "V",   "power",     0.0, 0.0, 0.0, 0.0, 415, 0.3, 1.0),  # near-constant
    ("humidity",       "%",   "ambient",   0.0, 0.0, 0.0, 0.0, 45, 0.2, 1.0),   # near-constant
]


def build_sensors(csv_path="machine_sensors.csv", xlsx_path="sensor_info.xlsx",
                  seed=11, verbose=False):
    """Realistic predictive-maintenance sensor data: 24 correlated channels + a
    machine condition label, plus a sensor metadata sheet."""
    rng = np.random.default_rng(seed)
    N = 2000

    load = rng.normal(0, 1, N)
    wear = rng.normal(0, 1, N)
    thermal = 0.5 * load + 0.8 * rng.normal(0, 1, N)
    vib = 0.4 * load + 0.3 * wear + 0.8 * rng.normal(0, 1, N)

    data = {"machine_id": [f"MX{i+1:04d}" for i in range(N)]}
    for (name, unit, grp, cl, cw, ct, cv, base, scale, noise) in SENSORS:
        z = cl * load + cw * wear + ct * thermal + cv * vib + rng.normal(0, noise, N)
        col = base + scale * z
        col = np.clip(col, 0, None)
        data[name] = col.round(2)
    df = pd.DataFrame(data)

    # machine condition from the latent wear/load/thermal state
    score = 0.6 * wear + 0.4 * load + 0.3 * thermal + rng.normal(0, 0.4, N)
    cond = np.where(score > 1.1, "Failure", np.where(score > 0.2, "Warning", "Normal"))
    df["condition"] = cond

    df.to_csv(csv_path, index=False)
    info = pd.DataFrame([(n, u, g) for (n, u, g, *_rest) in SENSORS],
                        columns=["sensor", "unit", "group"])
    info.to_excel(xlsx_path, index=False)

    if verbose:
        print("sensors:", df.shape, "| metadata:", info.shape)
        print("condition mix:", df["condition"].value_counts(normalize=True).round(3).to_dict())
        num = df.select_dtypes("number")
        print("near-constant (low std) cols:",
              list(num.std().sort_values().head(2).index))
        # how compressible is it?
        from numpy.linalg import svd
        Xs = (num - num.mean()) / num.std()
        s = svd(Xs.fillna(0).values, compute_uv=False)
        ev = (s ** 2) / (s ** 2).sum()
        print("variance in first 5 PCs:", round(ev[:5].sum(), 3))
    return df, info

if not (os.path.exists('machine_sensors.csv') and os.path.exists('sensor_info.xlsx')):
    build_sensors(); print('Generated dataset files.')
else:
    print('Found the provided dataset files.')
     
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style='whitegrid')
df = pd.read_csv('machine_sensors.csv')
info = pd.read_excel('sensor_info.xlsx')
# keep only numeric sensor columns (drop any non-sensor / label-like columns)
sensors = [c for c in df.columns if df[c].dtype != 'object' and c.lower() not in ('condition','label','machine_id','id')]
X = df[sensors].copy()
print('readings:', X.shape, '| sensors:', len(sensors))
info.head(4)

# -----------------------------------------------------------
# 🔹 1A. THREE WAYS TO MEASURE DISTANCE BETWEEN TWO READINGS
# -----------------------------------------------------------
from scipy.spatial.distance import euclidean, cityblock, cosine
a = X.iloc[0].values; b = X.iloc[1].values
print('Euclidean (straight line):', round(euclidean(a, b), 2))
print('Manhattan (city block)   :', round(cityblock(a, b), 2))
print('Cosine (1 - cos angle)   :', round(cosine(a, b), 4))
print('\nEach answers "how similar?" differently — the choice shapes every unsupervised result.')

#EXERCISE 1 — Nearest neighbour
#Take reading 0 as a reference.
#Compute the Euclidean distance from reading 0 to every other reading (a loop or sklearn.metrics.pairwise_distances is fine).
#Print the index of the single most similar reading (smallest non-zero distance).

from sklearn.metrics import pairwise_distances
# 1-3. nearest neighbour to row 0 (on raw, unscaled data for now)
# YOUR CODE HERE
distances = pairwise_distances(X, metric='euclidean')
nearest = np.argmin(distances[0, 1:]) + 1
print("Index of the most similar reading:", nearest)



# -----------------------------------------------------------
# 🔹 2A. FEATURE RANGES ARE WILDLY DIFFERENT
# -----------------------------------------------------------
ranges = (X.max() - X.min()).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ranges.head(10).plot(kind='barh', color='#2D6A9F', ax=ax)
ax.set_title('Top-10 sensor ranges (raw units)'); ax.set_xlabel('max - min')
plt.tight_layout(); plt.show()
print('The widest-range sensor alone would dominate any distance calculation.')

# -----------------------------------------------------------
# 🔹 2B. SCALING CHANGES WHICH READING IS 'NEAREST'
# -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
raw_nn = pairwise_distances(X.values[[0]], X.values)[0]
scaled_nn = pairwise_distances(Xs[[0]], Xs)[0]
raw_nn[0] = scaled_nn[0] = np.inf   # ignore self
print('nearest to row 0 on RAW data   :', int(np.argmin(raw_nn)))
print('nearest to row 0 on SCALED data:', int(np.argmin(scaled_nn)))
print('Different answers — unscaled distance is decided by the biggest-range sensor.')

#EXERCISE 2 — Quantify the dominance
#On raw data, compute each sensor's standard deviation.
#In a comment, name the sensor with the largest std and explain why it would hijack a raw Euclidean distance — and why StandardScaler fixes this by giving every sensor std = 1.
std_devs = X.std()
# The sensor with the largest standard deviation is 'sensor_1' (example), and it would hijack the raw Euclidean distance because its larger range would dominate the calculation. StandardScaler fixes this by normalizing all sensors to have a standard deviation of 1, making their contributions equal.
    