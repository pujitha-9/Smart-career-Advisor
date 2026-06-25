from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import os
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests  # For API call

# 🔑 Replace with your actual YouTube Data API key
YOUTUBE_API_KEY = "AIzaSyCx7-Yyjirf4w0Y_kc0HRCnR3Q3DTHyBEs"


def clean_input(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower().strip()


app = Flask(__name__)
app.secret_key = "super_secret_key"

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()

# Load Trained Model & Vectorizer
# Load Trained Model & Vectorizer
model_path = "career_model.pkl"
vectorizer_path = "career_mapping.pkl"  # Correct name here

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print("❌ Error loading model/vectorizer:", e)
        model, vectorizer = None, None
else:
    print("❌ Model files not found! Run train_model.py first.")
    model, vectorizer = None, None


# ✅ Career Descriptions Dictionary
career_descriptions = {
    "Marketing Manager": "Marketing Managers oversee campaigns, analyze trends, and develop strategies to promote products and services.",
    "AI Engineer": "AI Engineers build smart systems using machine learning and deep learning algorithms to mimic human intelligence.",
    "Data Scientist": "Data Scientists interpret complex data, build predictive models, and help organizations make informed decisions.",
    "UX Designer": "UX Designers enhance user satisfaction by improving usability, accessibility, and interaction in digital products.",
    "Software Developer": "Software Developers create applications and systems that solve problems and support user needs.",
    "Mechanical Engineer": "Mechanical Engineers design and test machines, engines, and tools to solve engineering problems.",
    "Content Writer": "Content Writers create engaging, informative, and persuasive content across digital and print platforms.",
    "Cybersecurity Analyst": "Cybersecurity Analysts protect systems and networks from cyber threats by monitoring, detecting, and responding to incidents.",
    "Business Analyst": "Business Analysts bridge the gap between IT and business using data analytics to assess processes and deliver insights.",
    "Financial Analyst": "Financial Analysts evaluate financial data and trends to guide investment and business decisions.",
    "Teacher": "Teachers educate students by delivering subject content, mentoring, and fostering development in schools and colleges.",
    "Web Developer": "Web Developers build and maintain websites, ensuring they are functional, user-friendly, and visually appealing.",
    "Graphic Designer": "Graphic Designers create visual content to communicate messages using typography, imagery, color, and layout.",
    "HR Manager": "HR Managers oversee recruiting, training, employee relations, and organizational development to support business goals.",
    "Project Manager": "Project Managers plan, execute, and close projects while managing teams, timelines, and budgets."
    # ➕ Add more as needed
}

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash("Username and Password are required!", "danger")
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials.", "danger")

    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('home'))

# Career Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        flash("Model missing. Train it first.", "danger")
        return redirect(url_for('home'))

    skills = request.form.get('skills', '').strip()
    interests = request.form.get('interests', '').strip()

    if not skills or not interests:
        flash("Please enter both professional skills and career interests!", "warning")
        return redirect(url_for('home'))

    user_input = skills + ", " + interests
    cleaned_input = clean_input(user_input)

    try:
        user_input_tfidf = vectorizer.transform([cleaned_input])
        predicted_career = model.predict(user_input_tfidf)[0]

        # ✅ Description fallback
        description = career_descriptions.get(predicted_career, "A promising career path with good growth potential.")

        # ✅ Placeholder API data
        demand_trend = 'High demand across industries'
        average_salary = '₹6-12 LPA (varies by location)'
        growth_rate = '7-10% annually'

        # ✅ External resource links
        search_term = predicted_career.replace(" ", "+")
        coursera_link = f"https://www.coursera.org/search?query={search_term}"
        udemy_link = f"https://www.udemy.com/courses/search/?q={search_term}"
        job_link = f"https://www.linkedin.com/jobs/search/?keywords={search_term}"

        # ✅ YouTube video search embed (dynamic for any prediction)
        video_id = get_youtube_video_id(predicted_career)
        youtube_embed = f"https://www.youtube.com/embed/{video_id}"  # Career Guidance for Students


        return render_template("result.html",
                               prediction=predicted_career,
                               description=description,
                               demand_trend=demand_trend,
                               average_salary=average_salary,
                               growth_rate=growth_rate,
                               coursera_link=coursera_link,
                               udemy_link=udemy_link,
                               job_link=job_link,
                               youtube_embed=youtube_embed)

    except Exception as e:
        print(f"❌ Error in Prediction: {e}")
        flash("Something went wrong during prediction.", "danger")
        return redirect(url_for('home'))
def get_youtube_video_id(query):
    career_video_map = {
        "AI Engineer": "JMUxmLyrhSk",               
        "Data Scientist": "ua-CiDNNj30",            
        "Web Developer": "Q33KBiDriJY",              
        "Cyber Security Analyst": "inWWhr5tnEA",     
        "Software Engineer": "rHux0gMZ3Eg",         
        "Mechanical Engineer": "4mZcVRLcy1Y",        
        "Graphic Designer": "5dLeN1i4iQo",         
        "Marketing Manager": "B5ZJ5z4fFHE",        
    }
    return career_video_map.get(query, "JMUxmLyrhSk")
if __name__ == "__main__":
    app.run(debug=True)
