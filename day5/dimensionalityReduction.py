# === SETUP: load the provided files (regenerate them if missing) ===
from ast import In
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
     

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style='whitegrid')

df = pd.read_csv('machine_sensors.csv')
try:
    info = pd.read_excel('sensor_info.xlsx')
except ImportError:
    info = pd.DataFrame([(n, u, g) for (n, u, g, *_rest) in SENSORS],
                        columns=['sensor', 'unit', 'group'])
    print("openpyxl is not installed; using in-memory sensor metadata fallback.")
sensors = [c for c in df.columns if c not in ('machine_id', 'condition')]
print('readings:', df.shape, '| sensors:', len(sensors))
df.head(3)

# -----------------------------------------------------------
# 🔹 1A. 24 SENSORS — but how many move together?
# -----------------------------------------------------------
corr = df[sensors].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax,
            xticklabels=True, yticklabels=True, cbar_kws={'shrink': .7})
ax.set_title('Sensor correlation matrix (lots of redundancy)')
plt.tight_layout(); plt.show()
     

# Count strongly-correlated sensor pairs (|r| > 0.8, excluding the diagonal)
c = corr.abs()
import numpy as np
pairs = (c.where(np.triu(np.ones(c.shape), k=1).astype(bool)) > 0.8).sum().sum()
print('sensor pairs with |correlation| > 0.8:', int(pairs))
print('=> many sensors carry overlapping information -> good candidate for reduction')

# -----------------------------------------------------------
# 🔹 1A. 24 SENSORS — but how many move together?
# -----------------------------------------------------------
corr = df[sensors].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax,
            xticklabels=True, yticklabels=True, cbar_kws={'shrink': .7})
ax.set_title('Sensor correlation matrix (lots of redundancy)')
plt.tight_layout(); plt.show()

# Count strongly-correlated sensor pairs (|r| > 0.8, excluding the diagonal)
c = corr.abs()
import numpy as np
pairs = (c.where(np.triu(np.ones(c.shape), k=1).astype(bool)) > 0.8).sum().sum()
print('sensor pairs with |correlation| > 0.8:', int(pairs))
print('=> many sensors carry overlapping information -> good candidate for reduction')



#EXERCISE 1 — Size up the redundancy
#Print how many sensors there are and how many machine readings.
#Find the single pair of sensors with the highest absolute correlation (other than 1.0).
#In a comment, explain why feeding all 24 sensors to a model could cause problems.

# 1. counts
print('Number of sensors:', len(sensors))
print('Number of machine readings:', df.shape[0])

# Find the single pair of sensors with the highest absolute correlation (other than 1.0)
max_corr = corr.abs().max().max()
if max_corr < 1.0:
    print('Highest absolute correlation (excluding 1.0):', round(max_corr, 3))
else:
    print('No correlation found (excluding 1.0).')

# Explanation
print("Feeding all 24 sensors to a model could cause problems due to multicollinearity, which can lead to unstable and unreliable model coefficients.")

# 2. most-correlated pair  (hint: stack the corr matrix, drop self-pairs, sort)
# YOUR CODE HERE
corr_pairs = corr.abs().stack()
corr_pairs = corr_pairs[corr_pairs < 1.0]  # exclude self-p
sorted_pairs = corr_pairs.sort_values(ascending=False)
most_corr_pair = sorted_pairs.index[0]
print('Most correlated sensor pair:', most_corr_pair,
    'with correlation:', round(float(sorted_pairs.iloc[0]), 3))

# 3. Why 24 sensors is a problem: ...   (comment)
print("Feeding all 24 sensors to a model could cause problems due to multicollinearity, which can lead to unstable and unreliable model coefficients.")


# -----------------------------------------------------------
# 🔹 2A. LOW-VARIANCE FILTER (drop near-constant sensors)
# -----------------------------------------------------------
# Compare each sensor's coefficient of variation (std / mean): near-zero = barely changes.
cov = (df[sensors].std() / df[sensors].mean()).abs().sort_values()
print('Lowest variation (candidates to drop):')
print(cov.head(3).round(4))
near_constant = cov.head(2).index.tolist()
print('\nDropping near-constant sensors:', near_constant)

# 1. drop near-constant
reduced = df[sensors].drop(columns=near_constant)

# 2. find & drop one of each highly-correlated (>0.9) pair
#    hint: iterate columns, track ones to drop using the upper triangle of reduced.corr().abs()
# YOUR CODE HERE

to_drop = set()
for i in range(len(reduced.columns)):
    for j in range(i + 1, len(reduced.columns)):
        if abs(reduced.corr().iloc[i, j]) > 0.9:
            to_drop.add(reduced.columns[j])
print('Dropping highly-correlated sensors:', to_drop)
reduced = reduced.drop(columns=to_drop)

# 3. how many remain?
# YOUR CODE HERE
print('Remaining sensors after low-variance filter:', reduced.shape[1])

# -----------------------------------------------------------
# 🔹 3A. WHICH SENSORS SAY MOST ABOUT MACHINE CONDITION?
# -----------------------------------------------------------
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(df[sensors], df['condition'], random_state=0)
mi = pd.Series(mi, index=sensors).sort_values(ascending=False)
print('Top 8 sensors by mutual information with condition:')
print(mi.head(8).round(3))

# 1. bar chart of top-10 MI scores
# YOUR CODE HERE
plt.figure(figsize=(8, 5))
sns.barplot(x=mi.head(10).values, y=mi.head(10).index, palette='viridis')
plt.title('Top 10 Sensors by Mutual Information with Condition')
plt.xlabel('Mutual Information Score')
plt.ylabel('Sensor')
plt.tight_layout()
plt.show()  
# 2. Which sensor group dominates? ...   (comment — use sensor_info to look up groups)
top_sensors = mi.head(10).index
top_groups = info.set_index('sensor').loc[top_sensors]['group'] 
print('Sensor groups in top 10:', top_groups.value_counts())
print("The 'thermal' group dominates the top 10 sensors, indicating that thermal sensors are most informative about machine condition.")


# -----------------------------------------------------------
# 🔹 4A. STANDARDISE (essential!) THEN FIT PCA
# -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[sensors].values
Xs = StandardScaler().fit_transform(X)      # PCA needs standardised inputs
pca = PCA().fit(Xs)
evr = pca.explained_variance_ratio_
print('Variance explained by the first 5 components:')
print((evr[:5] * 100).round(1))
print('First 5 components together:', round(evr[:5].sum() * 100, 1), '%')

# -----------------------------------------------------------
# 🔹 5A. CUMULATIVE EXPLAINED VARIANCE (the scree view)
# -----------------------------------------------------------
cum = np.cumsum(evr)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(cum) + 1), cum * 100, marker='o', color='#3AAFA9')
ax.axhline(90, color='#F4B942', ls='--', label='90% threshold')
ax.set_xlabel('number of components'); ax.set_ylabel('cumulative variance (%)')
ax.set_title('Cumulative explained variance'); ax.legend()
plt.tight_layout(); plt.show()

# 1. components needed for 90% variance
# YOUR CODE HERE
components_90 = np.argmax(cum >= 0.9) + 1
print('Number of components needed to explain 90% variance:', components_90)

# 2. report the compression (24 -> ?)
# YOUR CODE HERE
compression = 24 / components_90
print(f'Compression ratio: {compression:.2f} (from 24 sensors to {components_90} components)')

# -----------------------------------------------------------
# 🔹 6A. COMPRESS 24 SENSORS -> 2 COMPONENTS, THEN PLOT
# -----------------------------------------------------------
X2 = PCA(n_components=2).fit_transform(Xs)
colors = {'Normal': '#2D6A9F', 'Warning': '#F4B942', 'Failure': '#C0392B'}
fig, ax = plt.subplots(figsize=(7, 5))
for cond, col in colors.items():
    m = df['condition'] == cond
    ax.scatter(X2[m, 0], X2[m, 1], s=12, alpha=0.6, color=col, label=cond)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(title='condition')
ax.set_title('24 sensors compressed to 2 PCA components')
plt.tight_layout(); plt.show()

# 1. Do the conditions separate? ...   (comment)
print("The conditions show some separation in the PCA plot, with 'Normal' and 'Failure' conditions forming distinct clusters, while 'Warning' overlaps with both, indicating that PCA captures some of the variance related to machine condition.")

# 2. top-3 sensors loading on PC1
# YOUR CODE HERE
loadings = pd.Series(pca.components_[0], index=sensors).abs().sort_values(ascending=False)
print('Top 3 sensors loading on PC1:')
print(loadings.head(3).round(3))


# -----------------------------------------------------------
# 🔹 7A. QUICK CHECK — classify condition with ALL 24 vs a few PCs
# -----------------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
y = df['condition']

full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pcap = make_pipeline(StandardScaler(), PCA(n_components=6), LogisticRegression(max_iter=1000))
s_full = cross_val_score(full, X, y, cv=5).mean()
s_pca = cross_val_score(pcap, X, y, cv=5).mean()
print(f'Accuracy with all 24 sensors : {s_full:.3f}')
print(f'Accuracy with 6 PCA components: {s_pca:.3f}')
print('=> a quarter of the dimensions keeps almost all the predictive power')

# 1. accuracy vs number of PCA components
# YOUR CODE HERE
components = [1, 2, 3, 4, 5, 6, 8, 10]
accuracies = []
for n in components:
    pcap = make_pipeline(StandardScaler(), PCA(n_components=n), LogisticRegression(max_iter=1000))
    score = cross_val_score(pcap, X, y, cv=5).mean()
    accuracies.append(score)
plt.figure(figsize=(8, 5))
plt.plot(components, accuracies, marker='o', color='#3AAFA9')
plt.axhline(s_full, color='#F4B942', ls='--', label='full sensors')
plt.xlabel('number of PCA components'); plt.ylabel('accuracy')
plt.title('Accuracy vs number of PCA components')
plt.legend()
plt.tight_layout(); plt.show()

# 2. Fewest components that match full accuracy: ...   (comment)
for n, acc in zip(components, accuracies):
    if acc >= s_full:
        print(f'Fewest PCA components to match full accuracy: {n} components with accuracy {acc:.3f}')
        break

    
      