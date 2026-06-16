from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def investigate_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
	"""Return outlier counts per numeric feature using the IQR rule."""
	rows = []
	for col in numeric_cols:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		mask = (df[col] < lower) | (df[col] > upper)
		rows.append(
			{
				"feature": col,
				"outlier_count": int(mask.sum()),
				"outlier_pct": round(mask.mean() * 100, 2),
				"lower_bound": round(lower, 3),
				"upper_bound": round(upper, 3),
			}
		)
	return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def main() -> None:
	data_path = Path(__file__).with_name("loan_applications.csv")
	df = pd.read_csv(data_path)
	print("Loaded shape:", df.shape)

	# 1) Remove duplicate loan IDs
	dup_count = int(df.duplicated(subset=["loan_id"]).sum())
	if dup_count > 0:
		df = df.drop_duplicates(subset=["loan_id"], keep="first").reset_index(drop=True)
	print("Duplicate loan_id rows removed:", dup_count)
	print("Shape after de-dup:", df.shape)

	# Define target and features
	target_col = "default"
	categorical_cols = ["home_ownership", "loan_purpose", "region", "prior_default"]

	# Avoid ID leakage by excluding loan_id from model inputs
	feature_df = df.drop(columns=[target_col, "loan_id"])
	y = df[target_col]

	numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

	# 2) Investigate outliers before modeling
	outlier_report = investigate_outliers_iqr(feature_df, numeric_cols)
	print("\nTop outlier-heavy numeric features:")
	print(outlier_report.head(8).to_string(index=False))

	# 3) Stratified sampling for train/test split
	X_train, X_test, y_train, y_test = train_test_split(
		feature_df,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)
	print("\nTarget distribution (full):")
	print(y.value_counts(normalize=True).round(3).rename("ratio"))
	print("Target distribution (train):")
	print(y_train.value_counts(normalize=True).round(3).rename("ratio"))
	print("Target distribution (test):")
	print(y_test.value_counts(normalize=True).round(3).rename("ratio"))

	# 4) Handle missing values + encode categoricals + scale numerics
	# Scaling is included because Logistic Regression and SVM are scale-sensitive.
	numeric_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
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

	# 5) Train baseline models requested by feature constraints
	lr_model = Pipeline(
		steps=[
			("prep", preprocessor),
			("clf", LogisticRegression(max_iter=2000, random_state=42)),
		]
	)
	svm_model = Pipeline(
		steps=[
			("prep", preprocessor),
			("clf", SVC(kernel="rbf")),
		]
	)

	lr_model.fit(X_train, y_train)
	svm_model.fit(X_train, y_train)

	lr_pred = lr_model.predict(X_test)
	svm_pred = svm_model.predict(X_test)

	print("\nLogistic Regression Accuracy:", round(accuracy_score(y_test, lr_pred), 4))
	print(classification_report(y_test, lr_pred))

	print("SVM Accuracy:", round(accuracy_score(y_test, svm_pred), 4))
	print(classification_report(y_test, svm_pred))


if __name__ == "__main__":
	main()
