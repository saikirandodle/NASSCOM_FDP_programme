
# Core imports for the whole lab
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


pd.set_option('display.max_columns', None)
print('Setup complete. pandas', pd.__version__)

# -----------------------------------------------------------
# A SMALL, ALREADY-CLEAN DATASET (Part 1 did the cleaning)
# -----------------------------------------------------------
# Mixed columns: an unordered category (city), an ordered category
# (size), two numeric features on very different scales, and a target.
df = pd.DataFrame({
    'city':   ['pune', 'delhi', 'mumbai', 'pune', 'delhi', 'mumbai', 'pune', 'delhi'],
    'size':   ['small', 'large', 'medium', 'medium', 'small', 'large', 'large', 'small'],
    'age':    [25, 41, 33, 29, 52, 38, 46, 22],
    'income': [38000, 92000, 55000, 47000, 120000, 76000, 88000, 41000],
    'bought': [0, 1, 0, 0, 1, 1, 1, 0],   # target
})
df
print("Raw dataset:")
print(df)

# -----------------------------------------------------------
# 🔹 1A. ONE-HOT ENCODING (for UNORDERED categories)
# -----------------------------------------------------------

# 'city' has no natural order -> one 0/1 column per category
city_ohe = pd.get_dummies(df['city'], prefix='city').astype(int)
city_ohe
 
print(city_ohe)

# -----------------------------------------------------------
# 🔹 1B. LABEL / ORDINAL ENCODING (for ORDERED categories)
# -----------------------------------------------------------

# 'size' has a real order: small < medium < large -> map to 0,1,2
size_order = {'small': 0, 'medium': 1, 'large': 2}
df['size_code'] = df['size'].map(size_order)
df[['size', 'size_code']]


# 1. one-hot encode just the 'city' column of the whole df
ohe = OneHotEncoder(sparse_output=False, dtype=int)
city_ohe = ohe.fit_transform(df[['city']])
print("One-hot encoded 'city' column:")
print(city_ohe)

# 2. ordinal-encode 'size' (map with size_order)
df['size_code'] = df['size'].map(size_order)
print("\nOrdinal-encoded 'size' column:")
print(df[['size', 'size_code']])

# 3. Why one-hot is wrong for size / ordinal is wrong for city: ...   (comment)
# One-hot encoding 'size' would create separate columns for 'small', 'medium', 
# and 'large', losing the inherent order and making it harder for models to learn relationships.
# Ordinal encoding 'city' would assign arbitrary numeric values to cities, implying a false order
# that doesn't exist, which can mislead models into thinking one city is "greater" than another.

# -----------------------------------------------------------
# 🔹 2A. THE PROBLEM — FEATURES ON DIFFERENT SCALES
# -----------------------------------------------------------

# income (tens of thousands) dwarfs age (tens). A distance-based
# model would treat income as nearly all that matters.
print(df[['age', 'income']].describe().loc[['min', 'max', 'mean']])
     

# -----------------------------------------------------------
# 🔹 2B. STANDARDISATION (Z-score)  vs  NORMALISATION (Min-Max)
# -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler

num = df[['age', 'income']]

z = StandardScaler().fit_transform(num)        # mean 0, std 1
m = MinMaxScaler().fit_transform(num)          # range [0, 1]

print('Standardised (mean~0, std~1):')
print(pd.DataFrame(z, columns=['age', 'income']).round(2).head(3))
print('\nMin-Max (range 0..1):')
print(pd.DataFrame(m, columns=['age', 'income']).round(2).head(3))

income = df[['income']]   # 2D shape for sklearn

# 1. standardise -> check mean ~0, std ~1
z = StandardScaler().fit_transform(income)
print("Standardised 'income':")
print(pd.DataFrame(z, columns=['income']).agg(['mean', 'std']))

# 2. min-max scale -> check min 0, max 1
m = MinMaxScaler().fit_transform(income)
print("\nMin-max scaled 'income':")
print(pd.DataFrame(m, columns=['income']).agg(['min', 'max']))

# -----------------------------------------------------------
# 🔹 3A. SPLIT FIRST, THEN FIT ON TRAIN ONLY
# -----------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['age', 'income']]
y = df['bought']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # FIT + transform on train
X_test_s  = scaler.transform(X_test)        # only TRANSFORM on test
print('train rows:', X_train.shape[0], '| test rows:', X_test.shape[0])
print('scaler learned mean from TRAIN only:', scaler.mean_.round(1))

# 1a. WRONG: fit on ALL data, then split (leakage!)
wrong_mean = StandardScaler().fit(X).mean_

# 1b. RIGHT: fit on TRAIN only
right_mean = StandardScaler().fit(X_train).mean_

print('fit-on-all mean :', wrong_mean.round(1))
print('fit-on-train mean:', right_mean.round(1))

# 2. Why fitting on all data is a problem: ...   (write your answer)

# Fitting on all data allows the scaler to learn from the test set, which can lead to data leakage. This means that the model may perform better on the test set than it would in a real-world scenario, because it has indirectly "seen" the test data during training. This can give an overly optimistic
#estimate of the model's performance and may not generalize well to new, unseen data.


# -----------------------------------------------------------
# 🔹 4A. COMBINE & BIN EXISTING COLUMNS
# -----------------------------------------------------------

fe = df.copy()
# combine: income per year of age (a crude 'earning rate')
fe['income_per_age'] = (fe['income'] / fe['age']).round(0)
# bin: turn continuous age into life-stage buckets
fe['age_group'] = pd.cut(fe['age'], bins=[0, 30, 45, 100],
                         labels=['young', 'mid', 'senior'])
fe[['age', 'income', 'income_per_age', 'age_group']]



# -----------------------------------------------------------
# 🔹 4B. EXTRACT FROM A DATE
# -----------------------------------------------------------

dates = pd.to_datetime(['2024-01-06', '2024-03-15', '2024-07-21', '2024-12-25'])
d = pd.DataFrame({'date': dates})
d['day_of_week'] = d['date'].dt.day_name()   # Monday, Tuesday, ...
d['month']       = d['date'].dt.month
d['is_weekend']  = d['date'].dt.dayofweek >= 5
d


ex = df.copy()

# 1. high_earner flag (income > median)
ex['high_earner'] = ex['income'] > ex['income'].median()
print("High earner flag:")
print(ex[['income', 'high_earner']])

# 2. bin income into 3 buckets with pd.cut
ex['income_bin'] = pd.cut(ex['income'], bins=3, labels=['low', 'medium', 'high'])
print("\nIncome bins:")
print(ex[['income', 'income_bin']])

# 3. show income + the new columns
print("\nIncome with new features:")
print(ex[['income', 'high_earner', 'income_bin']])

# -----------------------------------------------------------
# 🔹 5A. ONE LEAK-FREE PIPELINE: PREPROCESS + MODEL
# -----------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num_cols = ['age', 'income']
cat_cols = ['city', 'size']

# scale the numbers, one-hot the categories — all in one object
pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])
pipe = Pipeline([('prep', pre), ('model', LogisticRegression(max_iter=1000))])
print(pipe)


# -----------------------------------------------------------
# 🔹 5B. FIT THE WHOLE THING IN ONE CALL
# -----------------------------------------------------------

X = df[num_cols + cat_cols]
y = df['bought']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

pipe.fit(X_train, y_train)        # preprocessing fitted on TRAIN only — no leakage
acc = pipe.score(X_test, y_test)  # transforms test with train-fitted steps
print('Test accuracy:', round(acc, 2))
print('(small toy dataset — the point is the leak-free workflow, not the score)')

from sklearn.preprocessing import MinMaxScaler

Xn = df[['age', 'income']]
yn = df['bought']

# 1. Pipeline: MinMaxScaler -> LogisticRegression
pre = ColumnTransformer([
    ('num', MinMaxScaler(), ['age', 'income'])
])
pipe = Pipeline([('prep', pre), ('model', LogisticRegression(max_iter=1000))])
print("Pipeline steps:")
print(pipe)

# -----------------------------------------------------------
# 🔹 5B. FIT THE WHOLE THING IN ONE CALL
# -----------------------------------------------------------

X = df[num_cols + cat_cols]
y = df['bought']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

pipe.fit(X_train, y_train)        # preprocessing fitted on TRAIN only — no leakage
acc = pipe.score(X_test, y_test)  # transforms test with train-fitted steps
print('Test accuracy:', round(acc, 2))
print('(small toy dataset — the point is the leak-free workflow, not the score)')

from sklearn.preprocessing import MinMaxScaler

Xn = df[['age', 'income']]
yn = df['bought']

# 1. Pipeline: MinMaxScaler -> LogisticRegression
pre = ColumnTransformer([
    ('num', MinMaxScaler(), ['age', 'income'])
])
pipe = Pipeline([('prep', pre), ('model', LogisticRegression(max_iter=1000))])
print("Pipeline steps:")
print(pipe)

# 2. split + fit on train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.25, random_state=0)
pipe.fit(X_train, y_train)

# 3. print test accuracy
acc = pipe.score(X_test, y_test)
print('Test accuracy:', round(acc, 2))