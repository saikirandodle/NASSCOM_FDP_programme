import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# ── 1. Load dataset ───────────────────────────────────────────────────────────
student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features
y = student_performance.data.targets

df = pd.concat([X, y], axis=1)
print("── Raw dataset ──")
print("Shape:", df.shape)
print(df.head())

# ── 2. Inspect data types & basic info ────────────────────────────────────────
print("\n── dtypes ──")
print(df.dtypes)
print("\n── describe (numeric) ──")
print(df.describe())

# ── 3. Check for missing values ───────────────────────────────────────────────
print("\n── Missing values per column ──")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

# ── 4. Drop duplicate rows ────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n── Duplicates removed: {before - len(df)} rows (kept {len(df)}) ──")

# ── 5. Standardise column names (lowercase, spaces → underscores) ─────────────
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[\s\-]+", "_", regex=True)
)
print("\n── Cleaned column names ──")
print(df.columns.tolist())

# ── 6. Fix data types ─────────────────────────────────────────────────────────
# Binary yes/no columns → bool
yes_no_cols = [c for c in df.columns if df[c].dropna().isin(["yes", "no"]).all()]
for col in yes_no_cols:
    df[col] = df[col].map({"yes": True, "no": False})

# Remaining str/object columns → category
obj_cols = df.select_dtypes(include="object").columns.tolist()
for col in obj_cols:
    df[col] = df[col].astype("category")

print(f"\n── Converted {len(yes_no_cols)} yes/no cols to bool, "
      f"{len(obj_cols)} object cols to category ──")

# ── 7. Handle outliers with IQR capping ───────────────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
capped = {}
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    if outliers:
        df[col] = df[col].clip(lower=lo, upper=hi)
        capped[col] = outliers
print("\n── Outliers capped (IQR method) ──")
print(capped if capped else "No outliers found.")

# ── 8. Final summary ──────────────────────────────────────────────────────────
print("\n── Cleaned dataset ──")
print("Shape:", df.shape)
print(df.dtypes)
print("\nSample:")
print(df.head())

# ── 9. Create graphs from cleaned data ───────────────────────────────────────
sns.set_theme(style="whitegrid")
plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

# Graph 1: Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["g3"], bins=20, kde=True, color="steelblue")
plt.title("Final Grade (G3) Distribution")
plt.xlabel("G3")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plots_dir / "g3_distribution.png", dpi=200)
plt.close()

# Graph 2: Mean final grade by study time
plt.figure(figsize=(8, 5))
study_summary = df.groupby("studytime", observed=True)["g3"].mean().reset_index()
sns.barplot(data=study_summary, x="studytime", y="g3", color="coral")
plt.title("Average G3 by Study Time")
plt.xlabel("Study Time")
plt.ylabel("Average G3")
plt.tight_layout()
plt.savefig(plots_dir / "avg_g3_by_studytime.png", dpi=200)
plt.close()

# Graph 3: Absences vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="g3", hue="school", alpha=0.7)
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.tight_layout()
plt.savefig(plots_dir / "absences_vs_g3.png", dpi=200)
plt.close()

# Graph 4: Numeric correlation heatmap
plt.figure(figsize=(11, 8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(plots_dir / "correlation_heatmap.png", dpi=200)
plt.close()

print("\n── Graphs saved to ──")
for p in sorted(plots_dir.glob("*.png")):
    print(p)

