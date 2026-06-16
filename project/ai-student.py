import kagglehub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
def main() -> None:
  
    print("Loading dataset...")
    df = pd.read_csv("AI_Student_Performance_50000_with_Errors.csv")
    print("Loaded shape:", df.shape)
    print(df.head())
    
  # 1) Remove duplicate with student ids
    dup_count = int(df.duplicated(subset=["Student_ID"]).sum())
    if dup_count > 0:
        df = df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
    print("Duplicate student_id rows removed:", dup_count)
    print("Shape after de-dup:", df.shape)

    # 2. missing per column
    print('Missing per column:')
    print(df.isna().sum())

    




if __name__ == "__main__":   main()