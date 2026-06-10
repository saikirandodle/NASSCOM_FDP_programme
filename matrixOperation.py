import numpy as np


# Real-world scenario:
# A cafe sells coffee, sandwich, and juice.
# Order 1: 2 coffees + 1 sandwich + 1 juice = 230
# Order 2: 1 coffee + 3 sandwiches + 2 juices = 400
# Order 3: 3 coffees + 2 sandwiches + 2 juices = 430

A = np.array([
	[2, 1, 1],
	[1, 3, 2],
	[3, 2, 2],
])

b = np.array([230, 400, 430])

solution = np.linalg.solve(A, b)

print("Coefficient matrix A:\n", A)
print("\nConstants vector b:\n", b)
print("\nSolution [coffee, sandwich, juice]:\n", solution)

verification = A @ solution
print("\nVerification A @ solution:\n", verification)

print("\nInterpretation:")
print(f"Coffee price: {solution[0]:.2f}")
print(f"Sandwich price: {solution[1]:.2f}")
print(f"Juice price: {solution[2]:.2f}")
print("The verification matches b, so the prices satisfy all three orders.")
