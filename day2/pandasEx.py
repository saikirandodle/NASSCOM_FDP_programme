import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid')   # nicer default plot styling
np.random.seed(42)                 # reproducible results
print('Setup complete. NumPy', np.__version__, '| Pandas', pd.__version__)


M = np.arange(1, 13).reshape(3, 4)   # shape (3, 4)
N = np.arange(1, 9).reshape(4, 2)    # shape (4, 2)
print('M:\n', M)
print('N:\n', N)
print(M@N)

# 2. Column-wise sum (axis=0) and row-wise mean (axis=1) of M
col_sum = np.sum(M, axis=0)
row_mean = np.mean(M, axis=1)
print("Column-wise sum of M:\n", col_sum)
print("Row-wise mean of M:\n", row_mean)

# 3. Index of the max value in each row of M (hint: argmax with axis=1)
max_indices = np.argmax(M, axis=1)
print("Index of max value in each row of M:\n", max_indices)


df = sns.load_dataset('titanic')

print('Shape:', df.shape)
df.head()