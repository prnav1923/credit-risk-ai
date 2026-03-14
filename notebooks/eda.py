import pandas as pd
import numpy as np

df = pd.read_csv('data/lending-club/accepted_2007_to_2018q4/accepted_2007_to_2018Q4.csv', low_memory=False)

# Shape, target distribution
print(df.shape)
print(df['loan_status'].value_counts())

# Keep relevant loans only
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['target'] = (df['loan_status'] == 'Charged Off').astype(int)

print(f"Default rate: {df['target'].mean():.2%}")  # expect ~20%