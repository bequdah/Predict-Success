# Student Success Prediction System 🎓

A complete Machine Learning project that generates synthetic student data, trains a predictive model, and deploys it using a FastAPI backend with a modern web interface.

## 📋 Project Overview
This project was developed for the **AI 350: Data Science** course at **Jordan University of Science and Technology (JUST)**. The goal is to predict whether a student will pass or fail a course based on specific academic behaviors and the local university grading system.

### Grading Logic (JUST System)
- **Coursework (60%):** Midterms, quizzes, and labs.
- **Final Exam (40%):** Simulated final exam performance.
- **Passing Threshold:** Total Score >= 60/100.

---

## 🚀 Features
- **Synthetic Data Generation:** Custom script to generate realistic, non-linear student data (200 samples).
- **Machine Learning Model:** Uses a `RandomForestClassifier` for robust predictions.
- **FastAPI Backend:** High-performance API serving the model predictions.
- **Mandatory Attendance Rule:** Integrated business logic that automatically fails students with attendance below 70%, overriding the model for academic compliance.
- **Modern UI:** Responsive, glassmorphism-style interface for real-time interaction.
- **Deployment Ready:** Configured for local execution and Railway cloud deployment.

---

## 📂 Project Structure
```text
deployment_assignment/
├── data/
│   └── data.csv          # Generated synthetic dataset
├── models/
│   └── model.pkl         # Trained Random Forest model
├── scripts/
│   ├── generate_data.py  # Advanced data generation logic
│   └── train_model.py    # Model training and evaluation script
├── static/
│   └── style.css         # Modern UI styling
├── templates/
│   └── index.html        # Main web interface
├── main.py               # FastAPI application entry point
├── requirements.txt      # Project dependencies
├── Procfile              # Railway deployment configuration
├── runtime.txt           # Python version specification
└── README.md             # Project documentation
```

---

## 🛠️ Technical Implementation

### 1. Data Generation (`generate_data.py`)
Instead of a simple linear formula, we used realistic statistical distributions:
- **Study Hours:** Log-normal distribution (avg ~2.2h/day).
- **Attendance:** Beta distribution (mostly high attendance).
- **Coursework:** Normal distribution centered around 40/60.
- **Logic:** Includes interactions (e.g., high study + high attendance synergy) and noise (realistic outliers) to prevent a trivial 100% accuracy.

### 2. Model Training (`train_model.py`)
- **Algorithm:** Random Forest Classifier (100 estimators).
- **Split:** 80% Training / 20% Testing.
- **Evaluation:** Achieved ~62-65% accuracy, which is realistic for complex social/academic data.

### 3. API & UI (`main.py` & Frontend)
- **Framework:** FastAPI with Pydantic for data validation.
- **UI:** A modern frontend that communicates with the `/predict` endpoint using Fetch API.

---

## 💻 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data & Train Model:**
   ```bash
   python scripts/generate_data.py
   python scripts/train_model.py
   ```

3. **Start the Server:**
   ```bash
   uvicorn main:app --reload
   ```
4. **Open in Browser:** `http://127.0.0.1:8000`

---

## ☁️ Deployment (Railway)
The project is configured for [Railway](https://railway.app/).
- Simply push to GitHub and connect the repository to Railway.
- It will automatically use the `Procfile` and `runtime.txt`.

---

**Developed by:** [Your Name / bequdah]
**Instructor:** Dr. Malak Abdullah
