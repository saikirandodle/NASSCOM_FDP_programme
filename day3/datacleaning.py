
# Core imports for the whole lab
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
print('Setup complete. pandas', pd.__version__)


# -----------------------------------------------------------
# A DELIBERATELY MESSY DATASET (so the lab is self-contained)
# -----------------------------------------------------------
# Problems baked in: missing values, disguised missing ('N/A', -1),
# duplicate rows, a number stored as text, a date as text,
# an extreme outlier, and inconsistent city spellings.
raw = pd.DataFrame({
    'id':    [1, 2, 3, 4, 5, 6, 7, 7],
    'name':  ['Ana', 'Bo', 'Cy', 'Di', 'Eve', 'Fin', 'Gus', 'Gus'],
    'age':   [30, 25, np.nan, 41, -1, 38, 29, 29],
    'city':  [' Pune ', 'pune', 'DELHI', 'Delhi ', 'Mumbai', 'bombay', 'Pune.', 'Pune.'],
    'spend': ['120.5', '80.0', '200.2', 'N/A', '150.0', '99000', '110.0', '110.0'],
    'date':  ['2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
              '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-11'],
})
raw

# -----------------------------------------------------------
# 🔹 1A. A FEW COMMANDS REVEAL MOST PROBLEMS
# -----------------------------------------------------------

df = raw.copy()        # always work on a copy
df.info()              # types + non-null counts
 
 # -----------------------------------------------------------
# 🔹 1B. MISSING COUNTS, DUPLICATES & RANGES
# -----------------------------------------------------------

print('Missing per column:')
print(df.isna().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nNote: spend is type', df['spend'].dtype, "-> stored as text!")

# 1. duplicate row count
# YOUR CODE HERE

print('\nDuplicate rows:', df.duplicated().sum())


# 2. missing per column
# YOUR CODE HERE
print('Missing per column:')
print(df.isna().sum())

# -----------------------------------------------------------
# 🔹 2A. UNMASK DISGUISED MISSING VALUES
# -----------------------------------------------------------

# 'N/A' (in spend) and -1 (in age) are really missing -> make them NaN
df['spend'] = pd.to_numeric(df['spend'], errors='coerce')  # 'N/A' -> NaN, text -> number
df['age']   = df['age'].replace(-1, np.nan)                # sentinel -> NaN

print('Missing after unmasking:')
print(df[['age', 'spend']].isna().sum())


# -----------------------------------------------------------
# 🔹 2B. HANDLE THE GAPS (impute)
# -----------------------------------------------------------

# median is robust to skew/outliers -> good for 'spend' and 'age'
df['age']   = df['age'].fillna(df['age'].median())
df['spend'] = df['spend'].fillna(df['spend'].median())
print('Missing after imputing:', df[['age', 'spend']].isna().sum().sum())



ex = raw.copy()

# 1. unmask missing values (spend -> numeric, age -1 -> NaN)
# YOUR CODE HERE
ex['spend'] = pd.to_numeric(ex['spend'], errors='coerce')  # 'N/A' -> NaN, text -> number
ex['age']   = ex['age'].replace(-1, np.nan)                # sentinel -> NaN
print('Missing after unmasking:')
print(ex[['age', 'spend']].isna().sum())

# 2a. dropna version
# YOUR CODE HERE
ex_dropna = ex.dropna(subset=['age', 'spend'])
print('Missing after dropna:', ex_dropna[['age', 'spend']].isna().sum().sum())

# 2b. median-impute version
# YOUR CODE HERE
ex_impute = ex.copy()
ex_impute['age']   = ex_impute['age'].fillna(ex_impute['age'].median())
ex_impute['spend'] = ex_impute['spend'].fillna(ex_impute['spend'].median())
print('Missing after imputation:', ex_impute[['age', 'spend']].isna().sum().sum())

# 3. compare row counts
# YOUR CODE HERE
print('Original row count:', len(ex))
print('Row count after dropna:', len(ex_dropna))
print('Row count after imputation:', len(ex_impute))




# -----------------------------------------------------------
# 🔹 3A. DROP DUPLICATE ROWS
# -----------------------------------------------------------

print('Before:', df.shape)
df = df.drop_duplicates()
print('After :', df.shape, '-> removed the repeated Gus row')

# -----------------------------------------------------------
# 🔹 3B. FIX DATA TYPES
# -----------------------------------------------------------

# 'date' is text -> convert to real datetimes so sorting/maths work
df['date'] = pd.to_datetime(df['date'])
# 'city' is a category -> mark it as such (saves memory, signals intent)
df['city'] = df['city'].astype('string')
print(df.dtypes)


ex = raw.copy()

# 1. fix types: spend -> numeric, date -> datetime
# YOUR CODE HERE

ex['spend'] = pd.to_numeric(ex['spend'], errors='coerce')  # 'N/A' -> NaN, text -> number
ex['date'] = pd.to_datetime(ex['date'])
print(ex.dtypes)

# 2. drop duplicates
# YOUR CODE HERE
print('Before:', ex.shape)
ex = ex.drop_duplicates()   
print('After :', ex.shape, '-> removed the repeated Gus row')

# 3. dtypes + shape
# YOUR CODE HERE

# -----------------------------------------------------------
# 🔹 4A. THE IQR RULE
# -----------------------------------------------------------

# spend has a 99000 value among ~100s -> a clear outlier
q1, q3 = df['spend'].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f'Q1={q1:.1f}  Q3={q3:.1f}  IQR={iqr:.1f}')
print(f'Normal range: {low:.1f} to {high:.1f}')

outliers = df[(df['spend'] < low) | (df['spend'] > high)]
print('\nOutlier rows:')
print(outliers[['name', 'spend']])

# -----------------------------------------------------------
# 🔹 4B. ONE WAY TO TREAT THEM — CAP (winsorise)
# -----------------------------------------------------------

# clip values to the IQR bounds instead of deleting the row
df['spend_capped'] = df['spend'].clip(lower=low, upper=high)
print(df[['name', 'spend', 'spend_capped']])

# 1. Q1, Q3, IQR for 'age'
# YOUR CODE HERE

q1_age, q3_age = df['age'].quantile([0.25, 0.75])
iqr_age = q3_age - q1_age
print(f'Q1={q1_age:.1f}  Q3={q3_age:.1f}  IQR={iqr_age:.1f}')

# 2. lower & upper bounds
# YOUR CODE HERE
low_age, high_age = q1_age - 1.5 * iqr_age, q3_age + 1.5 * iqr_age
print(f'Normal range: {low_age:.1f} to {high_age:.1f}')

# 3. rows outside the bounds
# YOUR CODE HERE
outliers_age = df[(df['age'] < low_age) | (df['age'] > high_age)]
print('\nOutlier rows:')
print(outliers_age[['name', 'age']])



# -----------------------------------------------------------
# 🔹 5A. THE PROBLEM — ONE CITY, MANY SPELLINGS
# -----------------------------------------------------------

print(df['city'].value_counts())   # ' Pune ', 'pune', 'Pune.' all look different!

# -----------------------------------------------------------
# 🔹 5B. STANDARDISE THE STRINGS
# -----------------------------------------------------------

s = df['city'].astype('string')
s = s.str.strip()                       # trim whitespace
s = s.str.lower()                       # unify case
s = s.str.replace('.', '', regex=False) # drop stray punctuation
s = s.replace({'bombay': 'mumbai'})     # map known variants to one label
df['city'] = s
print(df['city'].value_counts())        # now clean categories

messy = pd.Series([' London ', 'london', 'LONDON', 'N.Y.', 'new york ', 'New York'],
                  dtype='string')

# 1. strip + lower
# YOUR CODE HERE
clean = messy.str.strip().str.lower()
print(clean)

# 2. map 'n.y.' -> 'new york'  (after lowering)
# YOUR CODE HERE
clean = clean.replace({'n.y.': 'new york'})
print(clean)

# 3. value_counts()
# YOUR CODE HERE
print(clean.value_counts())


# After all the steps above, here's the cleaned frame
clean = df.drop(columns=['spend_capped'])
print('Final shape:', clean.shape)
print('Missing values:', int(clean.isna().sum().sum()))
print('Duplicates    :', int(clean.duplicated().sum()))
clean
     