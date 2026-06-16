
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)

n = 50000

majors = ["STEM", "Business", "Medical", "Humanities"]
years = ["Freshman", "Sophomore", "Junior", "Senior"]
use_cases = ["Copywriting/Drafting", "Ideation", "Summarizing_Reading", "Debugging/Troubleshooting"]
skills = ["Beginner", "Intermediate", "Advanced"]
policies = ["Allowed_With_Citation", "Strict_Ban", "Allowed_With_Disclosure"]

df = pd.DataFrame({
    "Student_ID": np.arange(100001, 100001 + n),
    "Major_Category": np.random.choice(majors, n),
    "Year_of_Study": np.random.choice(years, n),
    "Pre_Semester_GPA": np.round(np.random.normal(3.2, 0.4, n).clip(0, 4), 3),
    "Weekly_GenAI_Hours": np.round(np.random.gamma(2, 5, n), 2),
    "Primary_Use_Case": np.random.choice(use_cases, n),
    "Prompt_Engineering_Skill": np.random.choice(skills, n, p=[0.4, 0.4, 0.2]),
    "Tool_Diversity": np.random.randint(1, 6, n),
    "Paid_Subscription": np.random.choice([True, False], n, p=[0.3, 0.7]),
    "Traditional_Study_Hours": np.round(np.random.normal(12, 4, n).clip(0, 40), 2),
    "Perceived_AI_Dependency": np.random.randint(1, 6, n),
    "Institutional_Policy": np.random.choice(policies, n),
    "Anxiety_Level_During_Exams": np.random.randint(1, 11, n),
    "Post_Semester_GPA": np.round(np.random.normal(3.25, 0.45, n).clip(0, 4), 3),
    "Skill_Retention_Score": np.round(np.random.normal(75, 15, n).clip(0, 100), 2),
    "Burnout_Risk_Level": np.random.choice(["Low", "Medium", "High"], n, p=[0.4, 0.4, 0.2])
})

# --------------------------
# Inject data quality issues
# --------------------------

# 1. Duplicates (~2%)
dup_rows = df.sample(frac=0.02, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)

# 2. Null values (~3% in selected columns)
for col in ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Prompt_Engineering_Skill",
            "Traditional_Study_Hours", "Skill_Retention_Score"]:
    idx = np.random.choice(df.index, int(len(df) * 0.03), replace=False)
    df.loc[idx, col] = np.nan

# 3. Outliers
outlier_idx = np.random.choice(df.index, 500, replace=False)
df.loc[outlier_idx[:200], "Weekly_GenAI_Hours"] = np.random.uniform(80, 150, 200)
df.loc[outlier_idx[200:350], "Traditional_Study_Hours"] = np.random.uniform(50, 100, 150)
df.loc[outlier_idx[350:], "Skill_Retention_Score"] = np.random.uniform(120, 200, 150)

# 4. Uppercase inconsistencies
upper_idx = np.random.choice(df.index, 1000, replace=False)
df.loc[upper_idx, "Major_Category"] = df.loc[upper_idx, "Major_Category"].str.upper()

upper_idx2 = np.random.choice(df.index, 1000, replace=False)
df.loc[upper_idx2, "Primary_Use_Case"] = df.loc[upper_idx2, "Primary_Use_Case"].str.upper()

# 5. Leading/trailing spaces
space_idx = np.random.choice(df.index, 500, replace=False)
df.loc[space_idx, "Year_of_Study"] = " " + df.loc[space_idx, "Year_of_Study"] + " "

# 6. Invalid categorical values
bad_idx = np.random.choice(df.index, 200, replace=False)
df.loc[bad_idx, "Burnout_Risk_Level"] = "Unknown"

# Save dataset
df.to_csv("AI_Student_Performance_50000_with_Errors.csv", index=False)

print(df.shape)
print("Dataset generated successfully.")
