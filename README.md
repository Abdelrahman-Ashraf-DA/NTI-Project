# ✈️ Airline Flight Delay Prediction & Analysis

An **End-to-End Data Science & Machine Learning** project built during the **National Telecommunication Institute (NTI)** training program (Advanced Data Analysis Track - Batch 9)[cite: 1, 2]. 

The project predicts whether an operating flight will arrive **15+ minutes late** based on departure-time operational metrics and provides interactive analytical dashboards[cite: 2].

---

## 📌 Project Overview
- **Business Goal:** Predict arrival delay risk to assist operational planning and improve customer satisfaction[cite: 2].
- **Dataset:** U.S. domestic flights dataset (Jan 2026) — **518,368 valid flight records** post-cleaning[cite: 2].
- **Class Imbalance:** 80% On-Time vs 20% Delayed[cite: 2].
- **Workflow:** Business Understanding ➔ Data Prep ➔ EDA ➔ Feature Engineering ➔ ML Modeling ➔ Evaluation ➔ App & BI Deployment[cite: 1, 2].

---

## 📊 Key Insights & EDA Findings
* **Departure Delay (`DEP_DELAY`):** The single strongest signal for arrival delays (Correlation ≈ 0.49)[cite: 2].
* **Time of Day:** Delay rates increase progressively throughout the day, peaking in the evening (6 PM – 12 AM at 26.11%) due to delay propagation[cite: 2].
* **Carriers:** Southwest Airlines had the highest on-time rate (85.1%), while JetBlue and Spirit experienced the highest delay rates (>28%)[cite: 2].
* **Distance:** Flight distance has almost no correlation with delays (≈ 0.03)[cite: 2].

---

## 🤖 Machine Learning Performance

We trained and evaluated multiple classification models after handling data leakage and feature scaling (StandardScaler)[cite: 2]:

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **92.19%** | **89.03%** | **71.10%** | **79.06%** | **Selected Final Model**[cite: 2] |
| Decision Tree (max_depth=8) | 92.22% | 90.57% | 69.74% | 78.80% | Evaluated[cite: 2] |

* **Selected Model:** **Logistic Regression** — Achieved the highest F1-Score (0.79 vs target 0.70) and optimal Recall for detecting operational delay risks[cite: 2].

---

## 👥 Team & Roles
* **Basel:** Business & Data Understanding (Phase 1 & 2)[cite: 1]
* **Muhanad:** Data Preparation & Cleaning Pipeline (Phase 3)[cite: 1]
* **Joseph:** Exploratory Data Analysis & Power BI Dashboard (Phase 4)[cite: 1]
* **Abdelrahman Ashraf:** Machine Learning Modeling, Evaluation & Local Streamlit Deployment (Phase 5, 6 & 7)[cite: 1, 2]

---

## 🛠️ Tools & Technologies
`Python` | `Pandas` | `NumPy` | `Scikit-Learn` | `Streamlit` | `Power BI` | `Joblib` | `Git/GitHub`[cite: 1, 2]
