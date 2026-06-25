import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample Career Data (More Options Added)
data = {
    "skills_interests": [
        "coding, python, data analysis, AI",
        "marketing, social media, advertising, sales",
        "healthcare, medicine, biology, nursing",
        "finance, accounting, investments, economics",
        "writing, journalism, content creation, editing",
        "graphic design, creativity, UI/UX, branding",
        "software engineering, cloud computing, DevOps",
        "mechanical engineering, physics, robotics",
        "law, legal consulting, corporate law",
        "teaching, education, mentoring, tutoring",
        "cybersecurity, ethical hacking, penetration testing",
        "civil engineering, architecture, construction",
        "entrepreneurship, business strategy, leadership",
        "game development, animation, 3D modeling",
        "automobile engineering, vehicle design, mechanics",
    ],
    "career": [
        "Data Scientist",
        "Marketing Manager",
        "Doctor",
        "Financial Analyst",
        "Journalist",
        "Graphic Designer",
        "Software Engineer",
        "Mechanical Engineer",
        "Lawyer",
        "Teacher",
        "Cybersecurity Analyst",
        "Civil Engineer",
        "Entrepreneur",
        "Game Developer",
        "Automobile Engineer",
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["skills_interests"])
y = df["career"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
with open("career_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("career_mapping.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model training complete! career_model.pkl and career_mapping.pkl are saved.")
