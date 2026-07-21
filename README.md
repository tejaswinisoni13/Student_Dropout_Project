# EduInsight - AI-Powered Student Performance & Dropout Risk Analysis

<p align="center">
  <img src="assets/banner.png" alt="EduInsight Banner" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

<p align="center">

An AI-powered Learning Analytics Dashboard that helps faculty identify student performance patterns, cluster students into learning groups, and analyze dropout risk using Machine Learning.

</p>

---

# Project Overview

EduInsight is a faculty-oriented analytics platform developed to transform educational data into actionable insights.

The system analyzes student academic records, attendance, quizzes, classroom engagement, and forum activities to identify learning patterns using multiple clustering algorithms.

Unlike traditional dashboards, EduInsight automatically compares different clustering techniques and provides meaningful visual analytics to support academic decision-making.

---

# Features

- Interactive Streamlit Dashboard
- CSV Upload Support
- Student Performance Analysis
- Dropout Risk Analysis
- K-Means Clustering
- Gaussian Mixture Model (GMM)
- Hierarchical Clustering
- Algorithm Comparison
- PCA Visualization
- Course Outcome Analysis
- Learning Outcome Analysis
- Performance Trends
- Interactive Charts
- Download Processed Results

---

# Dashboard Preview

## Home

<p align="center">
<img src="screenshots/home.png" width="90%">
</p>

---

## Overview

<p align="center">
<img src="screenshots/overview.png" width="90%">
</p>

---

## Performance Analysis

<p align="center">
<img src="screenshots/performance.png" width="90%">
</p>

---

## Cluster Analysis

<p align="center">
<img src="screenshots/clustering.png" width="90%">
</p>

---

## Course Outcomes

<p align="center">
<img src="screenshots/course_outcomes.png" width="90%">
</p>

---

# Live Demo

<p align="center">

<img src="screenshots/demo.gif" width="95%">

</p>

> Replace `demo.gif` with a short recording (20–30 seconds) of the application.

---

# Machine Learning Workflow

```text
                   Student Dataset (CSV)
                            │
                            ▼
                  Data Preprocessing
      ┌─────────────────────────────────┐
      │ Missing Value Handling          │
      │ Feature Scaling                 │
      │ Data Cleaning                   │
      └─────────────────────────────────┘
                            │
                            ▼
               Feature Engineering
                            │
                            ▼
               Unsupervised Learning
      ┌────────────┬────────────┬────────────┐
      │            │            │            │
      ▼            ▼            ▼
   K-Means       GMM      Hierarchical
      │            │            │
      └────────────┴────────────┘
                    │
                    ▼
          Cluster Label Generation
                    │
                    ▼
          Student Risk Categorization
                    │
                    ▼
         Interactive Analytics Dashboard
```

---

# Technologies Used

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Dashboard | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Scientific Computing | SciPy |

---

# Project Structure

```
EduInsight/
│
├── app.py
├── Student_Dropout_Project.ipynb
├── requirements.txt
├── README.md
│
├── dataset/
│      sample_dataset.csv
│
├── screenshots/
│      home.png
│      overview.png
│      clustering.png
│      performance.png
│      course_outcomes.png
│      demo.gif
│
├── models/
│      classifier.pkl
│      kmeans.pkl
│      gmm.pkl
│
├── docs/
│      Project_Report.pdf
│      Architecture.png
│
└── assets/
       banner.png
```

---

# Dataset Features

The model considers multiple academic and engagement features including:

- 10th Percentage
- 12th Percentage
- CGPA
- Average Marks
- Attendance
- Quiz Performance
- Prior Knowledge
- Resource Usage
- Forum Activity
- Student Engagement

---

# Machine Learning Algorithms

| Algorithm | Purpose |
|------------|----------|
| K-Means | Student Segmentation |
| Gaussian Mixture Model | Probabilistic Clustering |
| Hierarchical Clustering | Relationship Discovery |
| PCA | Visualization |
| StandardScaler | Feature Scaling |
| SimpleImputer | Missing Value Handling |

---

# Dashboard Modules

### Home

- Upload dataset
- Configure analysis
- Faculty overview

### Overview

- Student statistics
- Risk distribution
- Performance summary

### Learning Curve

- Quiz progression
- Pass vs Dropout analysis

### Performance

- Grade distribution
- Academic performance
- Boxplot analysis

### Cluster Analysis

- PCA Visualization
- Algorithm comparison
- Cluster quality metrics
- Dendrogram

### Course Outcomes

- CO attainment
- Learning outcome analysis

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EduInsight.git
```

Go inside the project

```bash
cd EduInsight
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run app.py
```

---

# Future Improvements

- Deep Learning based Prediction
- Real-Time Student Analytics
- Faculty Login
- Student Login
- Report Generation (PDF)
- Email Alerts
- Cloud Deployment
- Database Integration
- Role-Based Access
- Explainable AI (XAI)

---

# Results

The project successfully compares three clustering algorithms using:

- Silhouette Score
- Davies-Bouldin Index

The dashboard automatically identifies the best-performing clustering model for the uploaded dataset.

---

# Author

**Tejaswini Soni**

B.Tech Information Technology

SGSITS Indore

Machine Learning | Data Science | Full Stack Development

GitHub:
https://github.com/YOUR_USERNAME

LinkedIn:
https://linkedin.com/in/YOUR_LINKEDIN

---

# License

This project is licensed under the MIT License.

---

## Support

If you found this project useful, consider giving it a ⭐ on GitHub.
