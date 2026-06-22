from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def save_visualizations(df: pd.DataFrame, numeric_cols: list[str], target_col: str, output_dir: Path) -> None:
  # Ensure output directory exists before saving plots.
  output_dir.mkdir(parents=True, exist_ok=True)

  # Class distribution to inspect label balance.
  target_counts = df[target_col].value_counts().sort_values(ascending=False)
  plt.figure(figsize=(8, 5))
  target_counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"])
  plt.title("Burnout Risk Class Distribution")
  plt.xlabel(target_col)
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(output_dir / "target_distribution.png", dpi=150)
  plt.close()

  top_numeric = numeric_cols[:6]
  # Plot distributions for a small representative subset of numeric features.
  if top_numeric:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_numeric):
      axes[i].hist(df[col].dropna(), bins=30, color="#4c72b0", edgecolor="black", alpha=0.8)
      axes[i].set_title(f"Distribution: {col}")
      axes[i].set_xlabel(col)
      axes[i].set_ylabel("Frequency")
    for j in range(len(top_numeric), len(axes)):
      axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_distributions.png", dpi=150)
    plt.close(fig)

  corr_cols = [c for c in numeric_cols if df[c].notna().sum() > 0]
  # Heatmap helps quickly identify redundant or strongly related numeric features.
  if corr_cols:
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
  # Keep paths relative to this script so execution works from any working directory.
  data_path = Path(__file__).with_name("AI_Student_Performance_50000_with_Errors.csv")
  output_dir = Path(__file__).with_name("visualizations")

  print("Loading dataset...")
  df = pd.read_csv(data_path)
  print("Loaded shape:", df.shape)

  # Remove duplicate student records and keep the first occurrence.
  dup_count = int(df.duplicated(subset=["Student_ID"]).sum())
  if dup_count > 0:
    df = df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
  print("Duplicate Student_ID rows removed:", dup_count)
  print("Shape after de-dup:", df.shape)

  print("\nMissing values per column:")
  print(df.isna().sum())

  # Drop completely empty rows and enforce non-null target values.
  df = df.dropna(how="all")
  print("\nShape after dropping rows with all nulls:", df.shape)

  target_col = "Burnout_Risk_Level"
  df = df.dropna(subset=[target_col]).copy()
  print("Shape after dropping rows with null target:", df.shape)

  # Standardize text columns and normalize empty strings to missing values.
  text_cols = df.select_dtypes(include=["object", "string"]).columns
  for col in text_cols:
    cleaned = df[col].astype("string").str.strip()
    cleaned = cleaned.replace("", np.nan)
    as_object = cleaned.astype("object")
    df[col] = as_object.where(pd.notna(as_object), np.nan)

  # Remove noisy label values that are not valid learning targets.
  unknown_before = int((df[target_col].str.lower() == "unknown").sum())
  if unknown_before > 0:
    df = df[df[target_col].str.lower() != "unknown"].copy()
  print("Unknown target rows removed:", unknown_before)

  # Feature engineering: measure semester GPA change and AI-to-study ratio.
  df["GPA_Improvement"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
  df["AI_Study_Ratio"] = df["Weekly_GenAI_Hours"] / (df["Traditional_Study_Hours"] + 1)

  # GPA impact summary after AI usage.
  improved_count = int((df["GPA_Improvement"] > 0).sum())
  valid_gpa_rows = int(df["GPA_Improvement"].notna().sum())
  improvement_pct = (improved_count * 100 / valid_gpa_rows) if valid_gpa_rows else 0.0
  avg_gpa_change = float(df["GPA_Improvement"].mean()) if valid_gpa_rows else 0.0
  avg_prev_gpa = float(df["Pre_Semester_GPA"].mean()) if valid_gpa_rows else 0.0
  avg_curr_gpa = float(df["Post_Semester_GPA"].mean()) if valid_gpa_rows else 0.0

  print("\nGPA impact after AI usage:")
  print("Students with GPA increase:", improved_count)
  print("Increase percentage:", round(improvement_pct, 2), "%")
  print("Average previous GPA:", round(avg_prev_gpa, 3))
  print("Average current GPA:", round(avg_curr_gpa, 3))
  print("Average GPA change:", round(avg_gpa_change, 3))

  # Compare GPA change across AI usage bands.
  ai_hours_bands = pd.qcut(df["Weekly_GenAI_Hours"], q=4, duplicates="drop")
  band_summary = (
    df.assign(AI_Hours_Band=ai_hours_bands)
    .groupby("AI_Hours_Band", observed=True)
    .agg(
      previous_gpa=("Pre_Semester_GPA", "mean"),
      current_gpa=("Post_Semester_GPA", "mean"),
      gpa_change=("GPA_Improvement", "mean"),
      median_change=("GPA_Improvement", "median"),
      count=("GPA_Improvement", "count"),
    )
    .round(3)
  )
  print("\nGPA summary by AI usage band (previous vs current):")
  print(band_summary)

  # Visualize previous GPA vs current GPA by AI usage band and overlay GPA change.
  band_labels = [str(interval) for interval in band_summary.index]
  x_pos = np.arange(len(band_labels))
  bar_width = 0.36

  fig, ax1 = plt.subplots(figsize=(11, 6))
  ax1.bar(
    x_pos - bar_width / 2,
    band_summary["previous_gpa"],
    width=bar_width,
    color="#1f77b4",
    edgecolor="black",
    alpha=0.9,
    label="Previous GPA",
  )
  ax1.bar(
    x_pos + bar_width / 2,
    band_summary["current_gpa"],
    width=bar_width,
    color="#ff7f0e",
    edgecolor="black",
    alpha=0.9,
    label="Current GPA",
  )
  ax1.set_title("Previous vs Current GPA by AI Usage Band")
  ax1.set_xlabel("Weekly GenAI Hours Band")
  ax1.set_ylabel("Average GPA")
  ax1.set_xticks(x_pos)
  ax1.set_xticklabels(band_labels, rotation=20)

  ax2 = ax1.twinx()
  ax2.plot(
    x_pos,
    band_summary["gpa_change"],
    color="#2ca02c",
    marker="o",
    linewidth=2,
    label="GPA Change",
  )
  ax2.set_ylabel("Average GPA Change")
  ax2.axhline(0, color="gray", linewidth=1, linestyle="--")

  handles1, labels1 = ax1.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

  fig.tight_layout()
  fig.savefig(output_dir / "gpa_change_by_ai_hours_band.png", dpi=150)
  plt.close(fig)

  # Build feature matrix and target vector for modeling.
  feature_df = df.drop(columns=[target_col, "Student_ID"])
  y = df[target_col]

  # Split columns by data type for dedicated preprocessing pipelines.
  categorical_cols = feature_df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
  numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

  print("\nSaving matplotlib visualizations...")
  save_visualizations(df, numeric_cols, target_col, output_dir)
  print("Saved plots to:", output_dir)

  # Stratified split preserves class proportions in train and test sets.
  X_train, X_test, y_train, y_test = train_test_split(
    feature_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
  )

  # Numeric preprocessing: impute missing values, then scale.
  numeric_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )
  # Categorical preprocessing: impute mode, then one-hot encode.
  categorical_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_pipe, numeric_cols),
      ("cat", categorical_pipe, categorical_cols),
    ]
  )

  # Full modeling pipeline: preprocess -> select informative features -> classify.
  model = Pipeline(
    steps=[
      ("prep", preprocessor),
      ("select", SelectPercentile(score_func=f_classif, percentile=40)),
      ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ]
  )

  print("\nTraining model...")
  model.fit(X_train, y_train)

  # Predict on test data 
  y_pred = model.predict(X_test)

  print("\nModel accuracy:", round(accuracy_score(y_test, y_pred), 4))
  print("\nClassification report:")
  print(classification_report(y_test, y_pred, zero_division=0))

  # Retrieve transformed feature names and report which ones survived selection.
  all_feature_names = model.named_steps["prep"].get_feature_names_out()
  selected_mask = model.named_steps["select"].get_support()
  selected_features = all_feature_names[selected_mask]
  print(f"Selected features: {len(selected_features)} of {len(all_feature_names)}")
  print("Sample selected features:", selected_features[:15].tolist())


if __name__ == "__main__":
  main()
