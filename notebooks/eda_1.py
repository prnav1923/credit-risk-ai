import pandas as pd

df = pd.read_csv('data/lending-club/accepted_2007_to_2018q4/accepted_2007_to_2018Q4.csv', 
                  low_memory=False)

# Keep only our target loans
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['target'] = (df['loan_status'] == 'Charged Off').astype(int)

# Check correlation of ALL columns with target
corr = df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False)
print(corr.head(30))