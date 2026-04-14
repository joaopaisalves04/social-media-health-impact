# 📱 Social Media Health Impact

A machine learning project that predicts whether social media use has a **Negative**, **Neutral**, or **Positive** impact on a student's health — based on usage habits, sleep, mental health score, and platform.

## 🌐 Live App
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-media-health-impact.streamlit.app)

---

## 📊 Project Overview

| Step | Description |
|------|-------------|
| EDA | Distribution analysis, platform breakdowns, correlation heatmap |
| Preprocessing | Encoding, feature engineering (usage/sleep ratio, health×sleep) |
| Modelling | 7 baseline classifiers + hyperparameter tuning + soft voting ensemble |
| Explainability | SHAP summary, bar, dependence, and waterfall plots |

**Best model:** Gradient Boosting — **99.4% weighted F1** on the test set.

### Key Findings
- Students using social media **> 5h/day** are overwhelmingly in the Negative class
- **Mental Health Score** and **Daily Usage Hours** are the two strongest predictors
- **WhatsApp** users: 100% Negative impact in this dataset
- **LinkedIn** → most positive outcomes; **TikTok** → most negative

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/JoaoPaisAlves/social-media-health-impact.git
cd social-media-health-impact

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in this folder:
#    https://www.kaggle.com/datasets/sumeakash/impact-of-social-media-on-health
#    File: Social_media_impact_on_life.csv

# 4. Train the model (generates model_bundle.pkl)
python train_and_save_model.py

# 5. Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                            # Streamlit prediction app
├── train_and_save_model.py           # Training script → model_bundle.pkl
├── requirements.txt                  # Python dependencies
├── social_media_health_notebook.ipynb # Full analysis (EDA → SHAP)
└── .gitignore
```

> **Note:** `model_bundle.pkl` and the dataset CSV are excluded from the repo.  
> Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sumeakash/impact-of-social-media-on-health) and run `train_and_save_model.py` to generate the model.

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `SHAP` · `pandas` · `matplotlib` · `seaborn`

---

## 📄 Dataset

[Impact of Social Media on Health](https://www.kaggle.com/datasets/sumeakash/impact-of-social-media-on-health) — 1,705 students, 11 features.
