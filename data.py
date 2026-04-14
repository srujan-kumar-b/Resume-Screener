import pandas as pd
import re

# Load dataset
df = pd.read_csv("final_removed cloumns.csv")
df.fillna('', inplace=True)

# Combine text
df['text'] = df[['Education','Experience','Certifications','Achievements','Languages','Interests']].agg(' '.join, axis=1)

# Skills extraction
skills_list = ["python","java","sql","machine learning","html","css","javascript","react","aws","docker"]

df['skills_extracted'] = df['text'].apply(lambda t: [s for s in skills_list if s in t.lower()])

# Job description skills
jd_skills = ["python","sql","machine learning","aws"]

# Match score (fast version)
def simple_score(skills):
    return (len(set(skills) & set(jd_skills)) / len(jd_skills)) * 100

df['match_score'] = df['skills_extracted'].apply(simple_score)

# Predicted role (using existing category for hackathon)
df['predicted_role'] = df['Category']

# Missing skills
df['missing_skills'] = df['skills_extracted'].apply(lambda s: list(set(jd_skills) - set(s)))

# Final score
df['final_score'] = (df['match_score'] * 0.7) + (df['completeness_score'] * 0.3)

# Experience level
def exp_level(e):
    nums = re.findall(r'\d+', str(e))
    if nums:
        y = int(nums[0])
        if y < 1:
            return "Fresher"
        elif y < 3:
            return "Intermediate"
        else:
            return "Experienced"
    return "Unknown"

df['experience_level'] = df['Experience'].apply(exp_level)

# Resume gaps
def gaps(r):
    g = []
    if not r['Certifications']: g.append("No Certifications")
    if not r['Achievements']: g.append("No Achievements")
    if not r['Experience']: g.append("No Experience")
    return g

df['resume_gaps'] = df.apply(gaps, axis=1)

# Recommendations
def rec(r):
    out = []
    if r['missing_skills']:
        out.append("Learn " + ", ".join(r['missing_skills'][:2]))
    if "No Certifications" in r['resume_gaps']:
        out.append("Add certifications")
    if "No Achievements" in r['resume_gaps']:
        out.append("Add projects")
    return "; ".join(out)

df['recommendations'] = df.apply(rec, axis=1)

# Save final CSV
df.to_csv("final_resume_screener_dataset.csv", index=False)

print("✅ Final CSV created successfully!")