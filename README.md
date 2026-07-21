# EduInsight – AI-Powered Student Performance & Dropout Risk Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat-square&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=flat-square&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

EduInsight is an AI-powered Learning Analytics Dashboard developed to help educational institutions analyze student performance and identify students who may be at risk of dropping out.

The system applies multiple unsupervised machine learning algorithms to group students based on academic performance, attendance, assessment scores, and engagement. Faculty members can upload a dataset through an interactive Streamlit interface and receive meaningful insights supported by visual analytics.

---

## Key Features

- Interactive Streamlit dashboard
- CSV dataset upload
- Automated data preprocessing
- Missing value handling
- Feature scaling
- Student clustering using multiple algorithms
- Dropout risk categorization
- Interactive performance visualizations
- Learning and Course Outcome analysis
- Algorithm comparison using clustering metrics
- Download processed results

---

## Dashboard Modules

### Home
- Upload student dataset
- Configure clustering algorithm
- Select number of clusters
- Filter students based on risk level

### Overview
- Student summary
- Cluster distribution
- Performance statistics
- Attendance overview
- Risk distribution

### Learning Curve
- Quiz performance trend
- Pass vs dropout analysis
- Student progress visualization

### Performance Analysis
- Grade distribution
- Attendance analysis
- Feature comparison
- Academic performance metrics

### Cluster Analysis
- K-Means clustering
- Gaussian Mixture Model (GMM)
- Hierarchical clustering
- PCA visualization
- Dendrogram
- Algorithm comparison

### Course Outcomes
- Course Outcome attainment
- Learning Outcome mapping
- Academic performance summary

---

## Machine Learning Workflow

```text
Student Dataset
       │
       ▼
Data Preprocessing
       │
       ├── Missing Value Handling
       ├── Feature Scaling
       └── Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Machine Learning
       │
       ├── K-Means
       ├── Gaussian Mixture Model
       └── Hierarchical Clustering
       │
       ▼
Cluster Label Generation
       │
       ▼
Student Risk Classification
       │
       ▼
Interactive Dashboard
```

---

## Algorithms Used

| Algorithm | Purpose |
|------------|----------|
| K-Means | Student segmentation |
| Gaussian Mixture Model | Probabilistic clustering |
| Hierarchical Clustering | Relationship discovery |
| PCA | Dimensionality reduction |
| StandardScaler | Feature normalization |
| SimpleImputer | Missing value handling |

---

## Evaluation Metrics

The clustering models are evaluated using:

- Silhouette Score
- Davies-Bouldin Index

These metrics are used to compare clustering quality and identify the most suitable algorithm for the uploaded dataset.

---

## Dataset Features

The dashboard supports features such as:

- 10th Percentage
- 12th Percentage
- CGPA
- Average Marks
- Attendance
- Quiz Attempts
- Prior Knowledge
- Resource Usage
- Forum Replies
- Forum Views
- Student Engagement

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Dashboard | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Scientific Computing | SciPy |

---

## Project Structure

```text
EduInsight/
│
├── app.py
├── Student_Dropout_Project.ipynb
├── requirements.txt
├── README.md
│
├── dataset/
│   └── sample_dataset.csv
│
├── models/
│
├── docs/
│
└── assets/
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move to the project directory

```bash
cd EduInsight
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Project Highlights

- Faculty-oriented analytics dashboard
- Interactive and user-friendly interface
- Multiple clustering algorithms in a single application
- Comparative clustering analysis
- PCA-based visualization
- Course Outcome and Learning Outcome analysis
- Downloadable processed results
- Modular architecture for future enhancements

---

## Future Enhancements

- Supervised dropout prediction model
- Explainable AI (XAI)
- Student and faculty authentication
- Database integration
- PDF report generation
- Cloud deployment
- Real-time analytics
- Role-based access control

---

## License

This project is licensed under the MIT License.
