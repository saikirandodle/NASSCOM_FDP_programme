
# -----------------------------------------------------------
# 🧳 PRACTICAL: Getting Started with matplotlib Library
# -----------------------------------------------------------

import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
	sys.path.remove(script_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# 🔹 1. LINE PLOT
# -----------------------------------------------------------

# Create a simple line plot for sin(x)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Line Plot of sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------------------------------------
# 🔹 2. BAR CHART
# -----------------------------------------------------------

# Simple bar chart with categories and values
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.figure(figsize=(6, 4))
plt.bar(categories, values, color='orange')
plt.title('Bar Chart Example')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()



# -----------------------------------------------------------
# 🔹 6. PIE CHART
# -----------------------------------------------------------

# Pie chart to show proportions of fruit types
sizes = [30, 40, 15, 15]
labels = ['Apple', 'Banana', 'Mango', 'Grapes']
colors = ['red', 'yellow', 'orange', 'purple']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Fruits')
plt.axis('equal')
plt.show()
     