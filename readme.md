# AI-Powered Stroke Risk Prediction System
**Course:** Fundamentals of AI and ML (CSE3001)
**made by:** Janani B (25MIM10094) [VIT Bhopal University]
**Program:** Integrated M.Tech AI branch
 
## Project Overview
This project implements a hybrid diagnostic agent that predicts the likelihood of a stroke based on clinical parameters such as age, hypertension, and glucose levels. It combines **Supervised Machine Learning** with **Heuristic Logic** to ensure safety in clinical decision-making.

Developed a diagnostic tool to address stroke mortality. The system utilizes a Logistic Regression model trained on clinical data to provide risk percentages. To align with Module 2 (Knowledge Representation), a manual override layer was added to flag high-risk "critical" profiles (e.g., elderly patients with multiple comorbidities). The final model achieved a Binary ROC AUC score of 0.81, demonstrating high reliability in distinguishing risk factors.


## Technical Stack
- **Language:** Python 3.13
- **Framework:** Flask (Web Interface)
- **ML Libraries:** Scikit-Learn, Pandas, NumPy
- **Model:** Logistic Regression / Random Forest

## Structure
- `data/`: Healthcare dataset (Kaggle).
- `src/`: Core logic (`train.py` and `main_app.py`).
- `models/`: Serialized model files (`model.pkl`).

## Setup
1. Install dependencies: `pip install flask flask-cors pandas scikit-learn`
2. Train the model: `python src/train.py`
3. Run the app: `python src/main_app.py`