import pandas as pd

# Read skills from CSV
skills = pd.read_csv("skills.csv")

# Read resume text
with open("sample_resume.txt", "r") as file:
    resume = file.read().lower()

# Find matching skills
matched = []

for skill in skills["Skill"]:
    if skill.lower() in resume:
        matched.append(skill)

print("===== Resume Analyzer =====")
print("Skills Found:", matched)
print("Total Skills Found:", len(matched))
total_skills = len(skills)
matched_skills = len(matched)

ats_score = (matched_skills / total_skills) * 100

print("ATS Score:", round(ats_score, 2), "%")
print("\nMissing Skills:")

for skill in skills["Skill"]:
    if skill not in matched:
        print("-", skill)
        if ats_score >= 80:
                print("\nResume Status: Excellent")
    elif ats_score >= 60:
                print("\nResume Status: Good")
else:
                print("\nResume Status: Needs Improvement")
