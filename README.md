📘 KMRL Metro AI Scheduling & Risk Analytics Assistant

A Fleet Scheduling, Optimization, and Predictive Risk Analytics System for Metro Operations

🚆 Overview

The Metro AI Scheduling Assistant is an intelligent decision-support system designed for Kochi Metro Rail Limited (KMRL) and adaptable for any metro or railway fleet.
It automates daily train scheduling, predicts operational risks, and provides rich analytical insights through interactive dashboards.

This system integrates:

Machine Learning (risk prediction)

Operations Research (OR-Tools optimization engine)

Streamlit Dashboards (multi-page UI)

Data Analytics (fleet health insights)

Built with a modular and scalable architecture, the platform helps operational teams enhance reliability, safety, and efficiency.

🚀 Key Features
🔧 1. Automated Daily Train Scheduling

Uses Google OR-Tools (CP-SAT Solver) to generate optimized train schedules.

Respects operational constraints like:

Train availability

Maintenance windows

Risk limits

Assignment rules

Produces a clean, interactive schedule table.

🧠 2. ML-Based Risk Prediction

Predicts risk scores for each metro train.

Model trained on enhanced fleet datasets.

Includes explainability using SHAP.

Helps prioritize maintenance and reduce breakdown probabilities.

📊 3. Fleet Health & Analytics Dashboard

Real-time visualizations of fleet health indicators:

Risk distribution

Fleet performance metrics

Historical trends

Train-wise parameters

Designed for operations control rooms and engineering teams.

🧩 Project Structure
├── 1_-_Daily_Schedule.py
├── pages/
│   ├── Analytics_Dashboard.py
├── ml_model.py
├── optimizer.py
├── data_handler.py
├── data_loader.py
├── kmrl_enhanced_dataset.csv
├── kmrl_risk_model.joblib
├── shap_explainer.joblib
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_optimization_engine.ipynb
│   └── 04_save_model.ipynb
└── README.md

🛠️ Technologies Used

Python 3.11

Streamlit – UI and interactive dashboards

Pandas, NumPy – Data processing

OR-Tools (CP-SAT) – Optimization & scheduling engine

Scikit-Learn – ML model

SHAP – Explainability

Matplotlib / Plotly – Charts & analytics

📥 How to Run
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run 1_-_Daily_Schedule.py

📌 Use Cases

Metro rail scheduling

Predictive maintenance systems

Fleet health monitoring

Resource optimization

Control room decision dashboards

👨‍💻 Author

MD AL EMRAN
AI/ML Engineer | CSE-AIML Student
Chandigarh University
