import pandas as pd
import re

# Load dataset
df = pd.read_csv("final_resume_screener_dataset.csv", low_memory=False)

# Drop accidental export columns like Unnamed: 0, Unnamed: 1, etc.
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# -------------------------------
# 1. HANDLE MISSING VALUES
# -------------------------------
# Fill missing values with dtype-compatible defaults.
text_columns = df.select_dtypes(include=["object", "string"]).columns
numeric_columns = df.select_dtypes(include=["number"]).columns

df[text_columns] = df[text_columns].fillna("Not Mentioned")
df[numeric_columns] = df[numeric_columns].fillna(0)

# -------------------------------
# 2. REMOVE DUPLICATES (SAFE)
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# 3. STANDARDIZE TEXT (LOWERCASE)
# -------------------------------
text_cols = ['Education', 'Experience', 'Certifications',
             'Achievements', 'Languages', 'Interests']

for col in text_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# -------------------------------
# 4. CLEAN EXPERIENCE COLUMN
# -------------------------------
def clean_experience(exp):
    exp = str(exp).lower()
    numbers = re.findall(r'\d+', exp)
    return numbers[0] + " years" if numbers else "0 years"

df['Experience'] = df['Experience'].apply(clean_experience)

# -------------------------------
# 5. CLEAN LIST-TYPE COLUMNS
# -------------------------------
list_cols = ['skills_extracted', 'missing_skills', 'resume_gaps']

def clean_list(col):
    return col.apply(lambda x: x if isinstance(x, list) else str(x).replace("[","").replace("]","").replace("'","").split(", "))

for col in list_cols:
    df[col] = clean_list(df[col])

# -------------------------------
# 6. ENSURE NUMERIC COLUMNS
# -------------------------------
df['completeness_score'] = pd.to_numeric(df['completeness_score'], errors='coerce')
df['match_score'] = pd.to_numeric(df['match_score'], errors='coerce')
df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')

# -------------------------------
# 7. REMOVE EXTRA SPACES
# -------------------------------
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# -------------------------------
# 8. FINAL CHECK
# -------------------------------
print("Data Cleaning Completed")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_resume_screener_dataset.csv", index=False)
print("Saved: cleaned_resume_screener_dataset.csv")