# Core imports for the whole lab
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#np.random.seed(42)
#print('Setup complete. NumPy', np.__version__)


# -----------------------------------------------------------
# 🔹 1A. PROBABILITY BY SIMULATION
# -----------------------------------------------------------

# P(A) = favourable outcomes / total outcomes.
# We can estimate it by simulating many trials.
rolls = np.random.randint(1, 7, size=100_000)   # 100k dice rolls

p_even = (rolls % 2 == 0).mean()       # P(even)
p_gt4  = (rolls > 4).mean()            # P(roll > 4)
print('P(even) ~', round(p_even, 3), ' (true 0.5)')
print('P(>4)   ~', round(p_gt4, 3),  ' (true 0.333)')


# -----------------------------------------------------------
# 🔹 1B. THE ADDITION RULE (disjoint events)
# -----------------------------------------------------------

# Rolling a 1 OR a 2 — these events can't both happen (disjoint)
p_1 = (rolls == 1).mean()
p_2 = (rolls == 2).mean()
p_1_or_2 = (np.isin(rolls, [1, 2])).mean()
print('P(1) + P(2) =', round(p_1 + p_2, 3))
print('P(1 or 2)   =', round(p_1_or_2, 3), ' -> they match')

# LAB EXERCISE 1 — Probability by simulation
# Simulate flipping two coins 100,000 times (0 = tails, 1 = heads):
#
# Estimate P(both heads).
# Estimate P(at least one head).
# Check the addition idea: does P(0 heads) + P(1 head) + P(2 heads) sum to 1?

flips = np.random.randint(0, 2, size=(100_000, 2))   # 100k pairs of flips
heads = flips.sum(axis=1)   # number of heads in each pair: 0, 1 or 2

# 1. P(both heads)  -> heads == 2
p_both_heads = (heads == 2).mean()
# 2. P(at least one head) -> heads >= 1
p_at_least_one = (heads >= 1).mean()

# 3. Do P(0) + P(1) + P(2) sum to 1?
p_0_heads = (heads == 0).mean()
p_1_head = (heads == 1).mean()
p_2_heads = (heads == 2).mean()
print('P(0) + P(1) + P(2) =', round(p_0_heads + p_1_head + p_2_heads, 3))


# -----------------------------------------------------------
# 🔹 2A. CONDITIONAL PROBABILITY  P(A | B) = P(A and B) / P(B)
# -----------------------------------------------------------

rolls = np.random.randint(1, 7, size=100_000)

# P(roll is 6 | roll is even):  narrow the world to even rolls
even = rolls[rolls % 2 == 0]          # condition on 'even'
p_6_given_even = (even == 6).mean()
print('P(6 | even) ~', round(p_6_given_even, 3), ' (true 1/3)')



# -----------------------------------------------------------
# 🔹 2B. TESTING FOR INDEPENDENCE
# -----------------------------------------------------------

# Two events are independent if P(A|B) == P(A).
# 'roll > 3' and 'roll is even' -> are they independent?
A = rolls > 3
B = rolls % 2 == 0
p_A        = A.mean()
p_A_given_B = A[B].mean()
print('P(A)      =', round(p_A, 3))
print('P(A | B)  =', round(p_A_given_B, 3))
print('Independent?', np.isclose(p_A, p_A_given_B, atol=0.02))

# LAB EXERCISE 2 — Conditional probability
# Using fresh dice rolls:
#
# Estimate P(roll is odd | roll < 4).
# Estimate P(roll < 4) overall.
# Are the events 'roll is odd' and 'roll < 4' independent? Compare P(odd|<4) with P(odd).
#
# rolls = np.random.randint(1, 7, size=100_000)
#
# 1. P(odd | roll < 4)  -> condition on rolls < 4, then check odd

odd = rolls[rolls < 4]          # condition on 'roll < 4'
p_odd_given_lt4 = (odd % 2 == 1).mean()
print('P(odd | < 4) ~', round(p_odd_given_lt4, 3))

# 2. P(roll < 4) overall
p_lt4 = (rolls < 4).mean()
print('P(< 4) ~', round(p_lt4, 3))

# 3. Are the events independent?
p_odd = (rolls % 2 == 1).mean()
print('P(odd) ~', round(p_odd, 3))
print('Independent?', np.isclose(p_odd_given_lt4, p_odd, atol=0.02))

# 2. P(roll < 4) overall
p_lt4 = (rolls < 4).mean()
print('P(< 4) ~', round(p_lt4, 3))

# 3. Compare P(odd | <4) with P(odd) overall — independent?
p_odd = (rolls % 2 == 1).mean()
print('P(odd) ~', round(p_odd, 3))



