
import random

import numpy as np

print("--- Intermediate NumPy Exercise ---")
print("Complete each task by writing the requested NumPy code.")
print("-----------------------------------")

# Task 1: Array Creation Shortcuts
print("\n--- Task 1: Array Creation Shortcuts ---")

# 1.1 Create a 1D array named 'arr1_1' with numbers from 0 to 99 (inclusive) using arange.
arr1_1 = np.arange(100)

# 1.2 Create a 3x3 array named 'arr1_2' filled with ones.
arr1_2 = np.ones((3, 3))

#1.3 Create a 2x4 array named 'arr1_3' filled with the number 7.
arr1_3 = np.full((2, 4), 7)

#1.4 Create a 5x5 array named 'arr1_4' with random integers between 10 and 50 (inclusive).
arr1_4 = np.random.randint(10, 51, (5, 5))

print("arr1_1 (first 10 elements):", arr1_1[:10])
print("arr1_2:\n", arr1_2)
print("arr1_3:\n", arr1_3)
print("arr1_4 (first 2 rows):\n", arr1_4[:2])


# Task 2: Indexing and Slicing with Arrays
print("\n--- Task 2: Indexing and Slicing with Arrays ---")
data = np.array([[10, 20, 30, 40],
                 [50, 60, 70, 80],
                 [90, 100, 110, 120],
                 [130, 140, 150, 160]])


# 2.1 Extract the element at row 2, column 3 (0-indexed). Store it in 'val2_1'.
# 2.2 Extract the first row. Store it in 'row2_2'.
# 2.3 Extract the last column. Store it in 'col2_3'.
# 2.4 Extract the sub-array consisting of rows 1 and 2, and columns 0 and 1. Store it in 'sub_arr2_4'.
# 2.5 Using boolean indexing, select all elements in 'data' that are greater than 100. Store them in 'filtered_elements2_5'.

val2_1 = data[2, 3]
row2_2 = data[0, :]
col2_3 = data[:, -1]
sub_arr2_4 = data[0:2, 1:3]
filtered_elements2_5 = data[data > 100]
print("val2_1:", val2_1)
print("row2_2:", row2_2)
print("col2_3:", col2_3)
print("sub_arr2_4:\n", sub_arr2_4)
print("filtered_elements2_5:", filtered_elements2_5)

# Task 4: Aggregation Functions
print("\n--- Task 4: Aggregation Functions ---")
arr4_1 = np.array([[10, 5, 12],
                   [3, 8, 15],
                   [20, 7, 4]])


# 4.1 Calculate the sum of all elements in 'arr4_1'. Store it in 'sum4_1'.
# 4.2 Calculate the mean of each column. Store it in 'mean_cols4_2'.
# 4.3 Find the maximum value in each row. Store it in 'max_rows4_3'.
# 4.4 Calculate the standard deviation of the entire array. Store it in 'std4_4'.

sum4_1 = np.sum(arr4_1)
mean_cols4_2 = np.mean(arr4_1, axis=0)
max_rows4_3 = np.max(arr4_1, axis=1)
min_rows4_3 = np.min(arr4_1, axis=1)
std4_4 = np.std(arr4_1)

print("sum4_1:", sum4_1)
print("mean_cols4_2:", mean_cols4_2)
print("max_rows4_3:", max_rows4_3)
print("min_rows4_3:", min_rows4_3)
print("std4_4:", std4_4)

# Task 5: Reshaping and Flattening
print("\n--- Task 5: Reshaping and Flattening ---")
arr5_1 = np.arange(24) # 1D array from 0 to 23

# 5.1 Reshape 'arr5_1' into a 4x6 2D array. Store it in 'reshaped_arr5_2'.
# 5.2 Reshape 'arr5_1' into a 2x3x4 3D array. Store it in 'reshaped_arr5_3'.
# 5.3 Flatten 'reshaped_arr5_2' back into a 1D array. Store it in 'flattened_arr5_4'.
#     Use both `.ravel()` and `.flatten()` to see if there's a difference (though not visible in output).

reshaped_arr5_2 = arr5_1.reshape(4, 6)
reshaped_arr5_3 = arr5_1.reshape(2, 3, 4)
flattened_arr5_4_ravel = reshaped_arr5_2.ravel()
flattened_arr5_4_flatten = reshaped_arr5_2.flatten()
print("reshaped_arr5_2:\n", reshaped_arr5_2)
print("reshaped_arr5_3:\n", reshaped_arr5_3)
print("flattened_arr5_4 (ravel):", flattened_arr5_4_ravel)
print("flattened_arr5_4 (flatten):", flattened_arr5_4_flatten)


# Task 6: Broadcasting
print("\n--- Task 6: Broadcasting ---")
arr6_1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
scalar = 10
vector = np.array([100, 200, 300])

# 6.1 Add the 'scalar' to every element of 'arr6_1'. Store it in 'result6_1'.
# 6.2 Add the 'vector' to each row of 'arr6_1'. Store it in 'result6_2'.

result6_1 = arr6_1 + scalar
result6_2 = arr6_1 + vector
print("result6_1:\n", result6_1)
print("result6_2:\n", result6_2)

# Task 7: Copying vs. Viewing Arrays
print("\n--- Task 7: Copying vs. Viewing Arrays ---")
original_arr7 = np.array([1, 2, 3, 4, 5])

# 7.1 Create a 'view' of 'original_arr7' named 'view_arr7' by slicing the entire array.
#     Modify the first element of 'view_arr7' to 99.
#     Observe how 'original_arr7' changes.
# 7.2 Create a 'copy' of 'original_arr7' named 'copy_arr7' using `.copy()`.
#     Modify the last element of 'copy_arr7' to 0.
#     Observe how 'original_arr7' remains unchanged.

view_arr7 = original_arr7[:]
view_arr7[0] = 99
copy_arr7 = original_arr7.copy()

copy_arr7[-1] = 0
print("original_arr7 after modifying view_arr7:", original_arr7)
print("copy_arr7 after modification:", copy_arr7)


# Task 8: Stacking and Splitting Arrays
print("\n--- Task 8: Stacking and Splitting Arrays ---")
arr8_a = np.array([[1, 2], [3, 4]])
arr8_b = np.array([[5, 6], [7, 8]])
arr8_c = np.array([[9, 10]])

# 8.1 Vertically stack 'arr8_a' and 'arr8_b'. Store it in 'vstack_arr8_1'.
# 8.2 Horizontally stack 'arr8_a' and 'arr8_b'. Store it in 'hstack_arr8_2'.
# 8.3 Vertically stack 'vstack_arr8_1' and 'arr8_c'. Store it in 'combined_arr8_3'.
# 8.4 Split 'combined_arr8_3' into 3 equal parts (rows). Store them in 'part1', 'part2', 'part3'.

vstack_arr8_1 = np.vstack((arr8_a, arr8_b))
hstack_arr8_2 = np.hstack((arr8_a, arr8_b))
combined_arr8_3 = np.vstack((vstack_arr8_1, arr8_c))

# Task 9: Sorting and Searching
print("\n--- Task 9: Sorting and Searching ---")
arr9_1 = np.array([5, 2, 8, 1, 9, 4, 7, 3, 6])
arr9_2d = np.array([[30, 20, 10],
                    [60, 50, 40],
                    [90, 80, 70]])

# 9.1 Sort 'arr9_1' in ascending order. Store it in 'sorted_arr9_1'.
# 9.2 Sort 'arr9_2d' along columns (i.e., each column sorted independently). Store it in 'sorted_cols9_2'.
# 9.3 Find the indices where the value 9 is present in 'arr9_1'. Store it in 'idx9_3'. (Hint: use np.where)
# 9.4 Find the index of the maximum value in 'arr9_1'. Store it in 'max_idx9_4'.

sorted_arr9_1 = np.sort(arr9_1)
sorted_cols9_2 = np.sort(arr9_2d, axis=0)
idx9_3 = np.where(arr9_1 == 9)[0]
max_idx9_4 = np.argmax(arr9_1)
print("sorted_arr9_1:", sorted_arr9_1)
print("sorted_cols9_2:\n", sorted_cols9_2)
print("idx9_3:", idx9_3)
print("max_idx9_4:", max_idx9_4)
