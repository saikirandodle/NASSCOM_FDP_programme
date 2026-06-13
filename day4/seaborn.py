
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
	sys.path.remove(script_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Intermediate Matplotlib & Seaborn Exercise ---")
print("Complete each task by writing the requested plotting code.")
print("Focus on proper labeling, titles, and drawing insights.")
print("--------------------------------------------------")


# Load the built-in tips dataset
tips = sns.load_dataset("tips")

# Display the first 5 rows
print(tips.head())

# --- Task 1: Single Variable Distributions (Histogram, Pie Chart) ---


# 1.1 Histogram: Distribution of 'total_bill'
# Create a histogram of the 'total_bill' column.
# Add a title, x-label, and y-label. Use `bins=20`.
# Insight Question: What is the typical range of total bills? Is the distribution symmetric or skewed?


sns.histplot(data=tips, x='total_bill', bins=20)
plt.title('Distribution of Total Bills')
plt.xlabel('Total Bill ($)')
plt.ylabel('Frequency')
plt.show()


# 1.2 Pie Chart: Proportion of 'smoker' status
# Create a pie chart showing the proportion of smokers vs. non-smokers.
# Ensure the percentages are displayed on the slices.
# Insight Question: Is there a significant difference in the number of smokers vs. non-smokers in the dataset?
smoker_counts = tips['smoker'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
plt.title('Proportion of Smokers vs. Non-Smokers')
plt.axis('equal')
plt.show()

# --- Task 2: Relationships Between Two Variables (Scatter, Bar, Horizontal Bar) ---


# 2.1 Scatter Plot: 'total_bill' vs. 'tip'
# Create a scatter plot with 'total_bill' on the x-axis and 'tip' on the y-axis.
# Use 'smoker' as the `hue` to differentiate points by smoker status.
# Add a title and labels.
# Insight Question: Is there a relationship between total bill and tip amount? Does smoker status influence this relationship?
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='smoker')
plt.title('Scatter Plot of Total Bill vs. Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.legend(title='Smoker Status')
plt.show()

# 2.2 Bar Plot: Average 'total_bill' by 'day'
# Create a bar plot showing the average 'total_bill' for each 'day' of the week.
# Add error bars (which Seaborn's barplot does by default, showing confidence intervals).
# Add a title and labels.
# Insight Question: Which day has the highest average total bill? Are there noticeable differences between weekdays and weekends?


sns.barplot(data=tips, x='day', y='total_bill', ci='sd', palette='pastel')
plt.title('Average Total Bill by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Total Bill ($)')
plt.show()

# 2.3 Horizontal Bar Plot: Average 'tip' by 'time'
# Create a horizontal bar plot showing the average 'tip' for 'Lunch' vs. 'Dinner'.
# Add a title and labels.
# Insight Question: Do customers tip more during lunch or dinner?
sns.barplot(data=tips, y='time', x='tip', ci='sd', palette='muted')
plt.title('Average Tip by Time of Day')
plt.xlabel('Average Tip ($)')
plt.ylabel('Time of Day')
plt.show()

print("\n--- Exercise Complete! ---")
print("You've practiced creating various plots and interpreting their insights.")
print("Remember that good EDA often involves iterative plotting and questioning.")