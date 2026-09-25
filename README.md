# Student Performance Dashboard

An interactive Streamlit dashboard that uses **unsupervised K-Means clustering**
to group students into relative performance tiers — **High-Performing**,
**Moderate**, and **At-Risk** — from a class's academic or engagement data,
helping faculty quickly prioritize which students may need attention.

> Tiers are a relative, data-driven prioritization aid for faculty — **not** a
> calibrated dropout prediction or an official academic judgment.

## Features

- **Dual-schema support** — auto-detects and works with two different CSV
  formats out of the box:
  - **Academic records** (10th/12th score, CGPA, quizzes, attendance, forum activity)
  - **Classroom engagement logs** (raised hands, resources visited, discussion, parent involvement)
- **Automated preprocessing pipeline** — numeric validation, IQR-based outlier
  detection, winsorization, median imputation, standard scaling, and PCA(2D)
- **K-Means clustering** into 3 relative performance tiers, with cluster
  quality metrics (silhouette score, Davies-Bouldin index) and optional
  agreement metrics (ARI / NMI) against a dataset's own ground-truth label,
  when available
- **Interactive dashboard** with 5 views: Home, Overview, Learning Curve,
  Performance Analysis, and Course Outcomes (descriptive CO indicators)
- **Derived grade bands** (O / A / B / F) computed from each dataset's
  primary score
- **CSV export** of full results, including tier and risk level per student

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app UI
- [scikit-learn](https://scikit-learn.org/) — K-Means, PCA, scaling, imputation, metrics
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing

## Project Structure

```
.
├── app.py              # Streamlit dashboard (UI, charts, tabs)
├── pipeline.py         # Preprocessing + K-Means pipeline (schema-agnostic)
├── requirements.txt
├── data/                # Sample CSVs to try the app with
│   ├── Mtechlabeldataset_sample.csv
│   └── xAPI-Edu-Data_sample.csv
└── README.md
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Try it out

Upload a CSV from the `data/` folder (or your own file matching one of the
two supported schemas) using the sidebar uploader.

## Supported Data Schemas

The pipeline auto-detects which schema an uploaded CSV matches based on its
columns:

| Schema | Key columns | Score used |
|---|---|---|
| **mtech** | `10th Score(%)`, `12th Score(%)`, `CGPA(%)`, `AVG`, `QUIZ 1..6`, `Attendance`, `Resource Visited`, ... | `AVG` |
| **xapi** | `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`, `StudentAbsenceDays`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction` | Mean of the 4 native 0–100 engagement columns |

If a file doesn't match either schema closely enough, the app shows a clear
error naming the columns it expected.

## How the Pipeline Works

1. **Clean** column names and standardize the student ID column
2. **Detect** the dataset schema from the columns present
3. **Validate & convert** feature columns to numeric, flagging invalid cells
4. **Detect outliers** via the IQR method, then **winsorize** (clip at the
   1st/99th percentiles)
5. **Impute** missing values with the median, then **scale** with
   `StandardScaler`
6. **Reduce to 2D with PCA** (for visualization) and run **K-Means**
   (default K=3)
7. **Rank clusters** by a normalized composite feature score and label them
   High-Performing / Moderate / At-Risk
8. **Derive grade bands** (O ≥ 75, A 60–74.99, B 50–59.99, F < 50) from the
   primary score

## Disclaimer

This project uses unsupervised clustering on a single uploaded class's data.
Tiers, risk levels, and Course Outcome indicators are **descriptive and
relative to that class only** — they are not calibrated predictions,
diagnoses, or official institutional metrics.

## License

Add a license of your choice (e.g. MIT) before publishing, if you'd like
others to freely reuse this code.
