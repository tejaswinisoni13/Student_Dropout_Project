"""
EduLens - Student Performance & Dropout Analytics Dashboard (Streamlit frontend).

The backend (pipeline.py) is used unchanged. Everything shown here is computed
from the uploaded file and the pipeline's own outputs.
"""

import hashlib
import html
import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    CLUSTER_COLORS,
    MTECH_FEATURE_COLS,
    N_CLUSTERS,
    QUIZ_COLS,
    XAPI_BINARY_MAP,
    XAPI_FEATURE_COLS,
    XAPI_NUMERIC_COLS,
    add_derived_grade,
    clean_columns,
    detect_dataset_type,
    evaluate_k_values,
    preprocess,
    run_kmeans,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PRODUCT_NAME = "EduLens"
PRODUCT_TAG = "Student Performance & Dropout Analytics"

PRIMARY, PRIMARY_DARK = "#6366F1", "#4F46E5"
NAVY, NAVY2, MUTED = "#0F172A", "#1E293B", "#64748B"

TIERS = ["High-Performing", "Moderate", "At-Risk"]
GRADES = ["O", "A", "B", "F"]
GRADE_COLORS = {"O": "#7C3AED", "A": "#16A34A", "B": "#2563EB", "F": "#DC2626", "Unknown": "#94A3B8"}
GRADE_RULE = "O: score 75 or above · A: 60 to under 75 · B: 50 to under 60 · F: under 50"

MIN_RECORDS = 4          # K-Means with K = 3 needs more than 3 students
RECOMMENDED_RECORDS = 30
RECORD_RULE = (
    f"Minimum {MIN_RECORDS} student records (K-Means with K = {N_CLUSTERS} needs more than "
    f"{N_CLUSTERS}); {RECOMMENDED_RECORDS} or more recommended for stable groups."
)

PAGES = [
    "Upload Dataset", "Class Overview", "Student Performance", "At-Risk Students",
    "Cluster Analysis", "Model Evaluation", "Indicator Summary", "Student Details",
    "Export Reports",
]

DISPLAY = {
    "10th Score(%)": "10th score (%)", "12th Score(%)": "12th score (%)", "CGPA(%)": "CGPA (%)",
    "AVG": "AVG (quiz average)", "QUIZ Attempt": "Quiz attempts", "Attendance": "Attendance",
    "Aim/Objective": "Aim/Objective", "Prior Knowledge of subject": "Prior knowledge",
    "Resource Visited": "Resources visited", "No. of Replies Posted": "Forum replies posted",
    "No. of views on forum": "Forum views", "raisedhands": "Raised hands",
    "VisITedResources": "Resources visited", "AnnouncementsView": "Announcements viewed",
    "Discussion": "Discussion posts", "StudentAbsenceDays": "Absence over 7 days (1 = yes)",
    "ParentAnsweringSurvey": "Parent answered survey (1 = yes)",
    "ParentschoolSatisfaction": "Parent satisfied (1 = yes)",
}

SCHEMAS = {
    "mtech": {
        "name": "Academic records (M.Tech style)",
        "features": MTECH_FEATURE_COLS,
        "numeric": MTECH_FEATURE_COLS,
        "key": ["Attendance", "QUIZ Attempt"],
        "table": ["Attendance", "QUIZ Attempt", "CGPA(%)", "Resource Visited", "No. of views on forum"],
        "low_concern": ["Attendance", "QUIZ Attempt", "CGPA(%)", "Resource Visited",
                        "No. of views on forum", "No. of Replies Posted"],
        "high_concern": [],
        "score_note": "Derived score = the AVG column when present, otherwise the mean of all available indicators.",
    },
    "xapi": {
        "name": "Classroom engagement (xAPI style)",
        "features": XAPI_FEATURE_COLS,
        "numeric": XAPI_NUMERIC_COLS,
        "key": ["raisedhands", "VisITedResources"],
        "table": ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion", "StudentAbsenceDays"],
        "low_concern": ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"],
        "high_concern": ["StudentAbsenceDays"],
        "score_note": "Derived score = the mean of the four engagement counts (raised hands, resources visited, "
                      "announcements viewed, discussion).",
    },
}

CONSIDERATIONS = {
    "Attendance": "Check in about attendance and any barriers to attending.",
    "QUIZ Attempt": "Review whether quizzes were missed or blocked and whether the student can still attempt them.",
    "CGPA(%)": "Review prior academic standing; academic advising may be worth offering.",
    "StudentAbsenceDays": "Review the absence record and the reasons behind it.",
    "_engagement": "Low use of course activity; a personal check-in or pointers to course material may help.",
}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css(logged_in):
    hide_sidebar = "" if logged_in else (
        '[data-testid="stSidebar"], [data-testid="collapsedControl"], '
        '[data-testid="stSidebarCollapsedControl"] { display: none !important; }'
    )
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp td, .stApp th,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] {{
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}}
.stApp {{ background: #F8FAFC; color: {NAVY}; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.main .block-container {{ max-width: 1400px; padding: 1.6rem 2.2rem 3rem; }}
{hide_sidebar}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E2E8F0; }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] {{ padding-top: 1rem; }}

/* page header */
.page-head {{ margin: 0 0 1.2rem; }}
.page-eyebrow {{ color: {PRIMARY_DARK}; font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }}
.page-title {{ color: {NAVY}; font-size: 1.85rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.2; margin-top: .15rem; }}
.page-sub {{ color: {MUTED}; font-size: .92rem; margin-top: .35rem; max-width: 900px; line-height: 1.55; }}
.chip {{ display: inline-block; background: #EEF2FF; color: {PRIMARY_DARK}; border-radius: 999px; padding: .18rem .7rem; font-size: .72rem; font-weight: 600; margin: .55rem .4rem 0 0; }}

/* KPI cards */
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .9rem; margin: .4rem 0 1.2rem; }}
.kpi {{ background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.04); position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent, {PRIMARY}); }}
.kpi-label {{ color: {MUTED}; font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
.kpi-value {{ color: {NAVY}; font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; margin-top: .3rem; line-height: 1.1; }}
.kpi-sub {{ color: {MUTED}; font-size: .75rem; margin-top: .3rem; }}

/* callouts */
.callout {{ border-radius: 14px; padding: .85rem 1.1rem; margin: .4rem 0 1rem; font-size: .86rem; line-height: 1.6; border: 1px solid; }}
.callout b {{ font-weight: 700; }}
.callout.info {{ background: #EEF2FF; border-color: #C7D2FE; color: #312E81; }}
.callout.warn {{ background: #FFFBEB; border-color: #FDE68A; color: #78350F; }}
.callout.ok {{ background: #ECFDF5; border-color: #A7F3D0; color: #065F46; }}
.callout.err {{ background: #FEF2F2; border-color: #FECACA; color: #7F1D1D; }}
.callout.note {{ background: #F1F5F9; border-color: #E2E8F0; color: #334155; }}

/* generic card + bordered containers */
.card {{ background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.1rem 1.25rem; box-shadow: 0 1px 3px rgba(15,23,42,.05); margin-bottom: .9rem; }}
.card h4 {{ margin: 0 0 .35rem; color: {NAVY}; font-size: .95rem; font-weight: 700; }}
.card p, .card li {{ color: #475569; font-size: .84rem; line-height: 1.6; margin: 0; }}
.card ul {{ margin: .3rem 0 0 1rem; padding: 0; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}
.section-title {{ color: {NAVY}; font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 .5rem; }}
.empty {{ color: {MUTED}; text-align: center; padding: 2.2rem 1rem; font-size: .88rem; }}

/* badges */
.badge {{ display: inline-block; border-radius: 999px; padding: .15rem .65rem; font-size: .72rem; font-weight: 700; }}
.b-hp {{ background: #D1FAE5; color: #065F46; }} .b-mod {{ background: #FEF3C7; color: #92400E; }} .b-risk {{ background: #FEE2E2; color: #991B1B; }}

/* sidebar */
.brand {{ display: flex; align-items: center; gap: .7rem; margin-bottom: 1rem; }}
.brand-logo {{ width: 38px; height: 38px; border-radius: 11px; background: {PRIMARY}; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1rem; }}
.brand-name {{ font-weight: 800; font-size: 1.1rem; color: {NAVY}; line-height: 1.1; }}
.brand-tag {{ font-size: .66rem; color: {MUTED}; line-height: 1.25; }}
.side-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: .7rem .85rem; margin-bottom: .7rem; }}
.side-label {{ font-size: .65rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: {MUTED}; }}
.side-value {{ font-size: .86rem; font-weight: 600; color: {NAVY}; margin-top: .15rem; word-break: break-word; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .4rem; }}
[data-testid="stSidebar"] .stButton > button {{ justify-content: flex-start; text-align: left; border: 0; background: transparent; color: #334155; font-weight: 600; padding: .5rem .75rem; box-shadow: none; }}
[data-testid="stSidebar"] .stButton > button:hover {{ background: #EEF2FF; color: {PRIMARY_DARK}; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{ background: #EEF2FF; color: {PRIMARY_DARK}; }}
[data-testid="stSidebar"] .st-key-logout_btn button {{ border: 1px solid #E2E8F0; background: #fff; justify-content: center; color: #334155; }}

/* login */
.login-hero {{ background: {NAVY}; border-radius: 22px; padding: 2.6rem 2.4rem; min-height: 560px; box-shadow: 0 20px 50px rgba(15,23,42,.18); }}
.login-hero .brand-name {{ color: #FFFFFF !important; font-size: 1.35rem; }}
.login-hero .brand-tag {{ color: #94A3B8 !important; font-size: .72rem; }}
.login-title {{ color: #FFFFFF !important; font-size: 2rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.2; margin: 2.6rem 0 .8rem; }}
.login-desc {{ color: #CBD5E1 !important; font-size: .95rem; line-height: 1.65; }}
.login-point {{ color: #E2E8F0 !important; font-size: .88rem; margin-top: 1rem; padding-left: 1rem; border-left: 3px solid {PRIMARY}; line-height: 1.5; }}
.login-form-title {{ color: {NAVY}; font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; margin-top: 2rem; }}
.login-form-sub {{ color: {MUTED}; font-size: .88rem; margin: .3rem 0 1rem; }}

/* controls */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius: 10px; font-weight: 600; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {{ background: {PRIMARY}; border-color: {PRIMARY}; color: #fff; }}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{ background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; color: #fff; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E2E8F0; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; padding: .6rem .9rem; }}
.stTabs [aria-selected="true"] {{ color: {PRIMARY_DARK} !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: #fff; border: 2px dashed #C7D2FE; border-radius: 16px; padding: 1.6rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def page_header(step, title, sub, A=None):
    chips = ""
    if A is not None:
        chips = (f'<span class="chip">{esc(A["name"])}</span>'
                 f'<span class="chip">{A["n"]} students</span>'
                 f'<span class="chip">{esc(SCHEMAS[A["kind"]]["name"])}</span>')
    st.markdown(
        f'<div class="page-head"><div class="page-eyebrow">{esc(step)}</div>'
        f'<div class="page-title">{esc(title)}</div><div class="page-sub">{sub}</div>{chips}</div>',
        unsafe_allow_html=True,
    )


def kpis(items):
    """items: (label, value, sub, accent_hex)"""
    cards = "".join(
        f'<div class="kpi" style="--accent:{a}"><div class="kpi-label">{esc(l)}</div>'
        f'<div class="kpi-value">{esc(v)}</div><div class="kpi-sub">{esc(s)}</div></div>'
        for l, v, s, a in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def callout(kind, body):
    st.markdown(f'<div class="callout {kind}">{body}</div>', unsafe_allow_html=True)


def card(title, body_html):
    st.markdown(f'<div class="card"><h4>{esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{esc(text)}</div>', unsafe_allow_html=True)


def badge(tier):
    cls = {"High-Performing": "b-hp", "Moderate": "b-mod", "At-Risk": "b-risk"}.get(tier, "b-mod")
    return f'<span class="badge {cls}">{esc(tier)}</span>'


def label(col):
    return DISPLAY.get(col, col)


def pct(a, b):
    return f"{(a / b * 100):.1f}%" if b else "0.0%"


def goto(page):
    st.session_state["page"] = page


def safe_df(df):
    """Make mixed-type object columns Arrow-safe for display."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and str(pd.api.types.infer_dtype(out[c], skipna=True)).startswith("mixed"):
            out[c] = out[c].astype(str)
    return out


def show_df(df, height=None):
    kw = {"use_container_width": True, "hide_index": True}
    if height:
        kw["height"] = height
    st.dataframe(safe_df(df) if not hasattr(df, "style") else df, **kw)


def style_fig(fig, h=340, title=None, legend=False):
    fig.update_layout(
        template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        margin=dict(t=58 if title else 24, b=78 if legend else 44, l=52, r=24),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E2E8F0"),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0, title_text=""))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=NAVY)))
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, linecolor="#E2E8F0")
    return fig


def plot(fig, key=None):
    """Render a figure in a bordered card; None means there is nothing to draw."""
    with st.container(border=True):
        if fig is None:
            st.markdown('<div class="empty">Not enough data available to draw this chart.</div>',
                        unsafe_allow_html=True)
        else:
            st.plotly_chart(fig, use_container_width=True, key=key)


def has_values(df, col):
    return col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()


def fmt(v):
    if pd.isna(v):
        return "-"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------------------
# Data layer: upload validation, analysis, flags, exports
# ----------------------------------------------------------------------------
def read_csv_bytes(data):
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc), None
        except pd.errors.EmptyDataError:
            return None, "The file is empty."
        except Exception as exc:  # parser errors etc.
            last = exc
    return None, f"The file could not be read as CSV ({last})."


def to_numeric_like_pipeline(series, col, kind):
    """Numeric view of the *uploaded* values, using the same mapping as pipeline.py."""
    if kind == "xapi" and col in XAPI_BINARY_MAP:
        return series.map(XAPI_BINARY_MAP[col])
    return pd.to_numeric(series, errors="coerce")


def validate_upload(raw):
    v = {"errors": [], "warnings": [], "kind": None, "n": len(raw), "n_cols": raw.shape[1],
         "rows": [], "used": [], "missing_cells": 0, "dup_rows": 0, "dup_ids": 0}
    if raw.shape[1] == 0 or len(raw) == 0:
        v["errors"].append("The file contains no data rows or no columns.")
        return v
    cleaned = clean_columns(raw)
    kind = detect_dataset_type(cleaned)
    v["kind"] = kind
    if kind is None:
        v["errors"].append(
            "The columns do not match a supported dataset. The pipeline needs at least 3 recognised columns "
            "from one schema (see 'Supported datasets' above).")
        return v
    if v["n"] < MIN_RECORDS:
        v["errors"].append(f"Only {v['n']} records found. {RECORD_RULE}")
    elif v["n"] < RECOMMENDED_RECORDS:
        v["warnings"].append(f"Small cohort ({v['n']} records): groups can change a lot with a few students. "
                             f"{RECOMMENDED_RECORDS}+ recommended.")
    sch = SCHEMAS[kind]
    n = v["n"]
    for col in sch["features"]:
        found = col in cleaned.columns
        role = "Categorical (Yes/No style)" if (kind == "xapi" and col in XAPI_BINARY_MAP) else "Numeric"
        row = {"Column": col, "Role": role, "Found": "Yes" if found else "No", "Missing": "", "Invalid": ""}
        if found:
            s = cleaned[col]
            num = to_numeric_like_pipeline(s, col, kind)
            invalid = int((s.notna() & num.isna()).sum())
            missing = int(s.isna().sum())
            row["Missing"], row["Invalid"] = missing, invalid
            v["used"].append(col)
            v["missing_cells"] += missing
            if num.notna().sum() == 0:
                if kind == "xapi" and col in XAPI_BINARY_MAP:
                    allowed = " / ".join(XAPI_BINARY_MAP[col])
                    v["errors"].append(f"Column '{col}' has no valid values. Expected text values: {allowed}.")
                else:
                    v["errors"].append(f"Column '{col}' has no valid numeric values.")
            elif invalid:
                v["warnings"].append(
                    f"Column '{col}': {invalid} value(s) are not valid and will be treated as missing "
                    "(filled with the column median).")
        v["rows"].append(row)
    absent = [c for c in sch["features"] if c not in cleaned.columns]
    if absent:
        extra = ""
        if kind == "mtech" and "AVG" not in cleaned.columns:
            extra = " 'AVG' is missing, so the derived score will be the mean of the other indicators."
        v["warnings"].append(f"{len(absent)} recognised column(s) not in file and will be skipped: "
                             f"{', '.join(absent)}.{extra}")
    v["dup_rows"] = int(cleaned.duplicated().sum())
    if v["dup_rows"]:
        v["warnings"].append(f"{v['dup_rows']} fully duplicated row(s) found. They are kept and analysed as "
                             "separate students; remove them first if they are accidental.")
    if "Student ID" in cleaned.columns:
        v["dup_ids"] = int(cleaned["Student ID"].duplicated().sum())
        if v["dup_ids"]:
            v["warnings"].append(f"{v['dup_ids']} repeated student ID(s) found.")
    else:
        v["warnings"].append("No 'Student ID' or 'Roll Number' column. Row numbers 1 to N will be used as IDs.")
    if kind == "mtech":
        qmissing = [c for c in QUIZ_COLS if c not in cleaned.columns]
        if qmissing and len(qmissing) < len(QUIZ_COLS):
            v["warnings"].append(f"Quiz columns not found: {', '.join(qmissing)}.")
    return v


def build_analysis(name, raw):
    """Run the unchanged backend and assemble everything the pages need."""
    cleaned = clean_columns(raw).reset_index(drop=True)
    prep = preprocess(cleaned)
    df, _model, metrics = run_kmeans(prep, N_CLUSTERS)
    df = add_derived_grade(df, "primary_score")
    g = df["Grade"].astype(object)
    df["Grade"] = g.where(g.notna(), "Unknown").astype(str)
    df = df.reset_index(drop=True)

    kind, id_col, features = prep["dataset_type"], prep["id_col"], list(prep["available_cols"])
    n = len(df)
    orig = cleaned.copy()
    if id_col not in orig.columns:
        orig.insert(0, id_col, range(1, n + 1))
    quiz_cols = [c for c in prep["quiz_cols"] if c in df.columns]
    for c in quiz_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # per-indicator processing trail (original vs processed)
    trail = []
    for c in features:
        raw_num = to_numeric_like_pipeline(orig[c], c, kind)
        lo, hi = prep["clipping_bounds"][c]
        imputed = int(raw_num.isna().sum())
        clipped = int((raw_num.notna() & ((raw_num - df[c]).abs() > 1e-9)).sum())
        trail.append({"col": c, "imputed": imputed, "clipped": clipped, "lo": lo, "hi": hi,
                      "outliers": prep["outliers"].get(c, 0)})

    quart = {c: float(df[c].quantile(0.25)) for c in features}
    sch = SCHEMAS[kind]
    reasons = [[] for _ in range(n)]
    reason_cols = [[] for _ in range(n)]
    for c in sch["low_concern"]:
        if c in df.columns:
            for i in np.where(df[c].values < quart[c])[0]:
                reasons[i].append(f"{label(c)} below class lower quartile ({fmt(df[c].iat[i])} vs {fmt(quart[c])})")
                reason_cols[i].append(c)
    for c in sch["high_concern"]:
        if c in df.columns:
            for i in np.where(df[c].values >= 1)[0]:
                reasons[i].append("More than 7 absence days recorded")
                reason_cols[i].append(c)
    df["Reasons"] = ["; ".join(r) if r else "No indicator below the class lower quartile" for r in reasons]
    df["_reason_cols"] = reason_cols
    df["_n_reasons"] = [len(r) for r in reasons]

    is_risk, is_f = df["cluster"] == "At-Risk", df["Grade"] == "F"
    df["Flag basis"] = np.select(
        [is_risk & is_f, is_risk, is_f], ["At-Risk tier + Grade F", "At-Risk tier only", "Grade F only"], "")
    df["In intervention list"] = is_risk | is_f
    df["Priority"] = np.select([is_risk & is_f, is_risk | is_f], [0, 1], 2)

    return {
        "name": name, "n": n, "kind": kind, "id_col": id_col, "features": features,
        "score_col": "primary_score", "score_label": prep["score_label"], "quiz_cols": quiz_cols,
        "df": df, "orig": orig, "metrics": metrics, "pca_variance": prep["pca_variance"],
        "X_scaled": prep["X_scaled"], "trail": trail, "quartiles": quart, "k_eval": None,
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def results_table(A, basis, only_flagged=False):
    """basis: 'original' (as uploaded) or 'processed' (winsorized + imputed values used by the model)."""
    df = A["df"]
    src = A["orig"] if basis == "original" else df
    out = pd.DataFrame({A["id_col"]: df[A["id_col"]].values})
    for c in A["features"]:
        out[c] = src[c].values
    for c in A["quiz_cols"]:
        out[c] = df[c].values          # quiz columns are never altered by the pipeline
    out["Derived_Score_(from_processed_values)"] = df[A["score_col"]].round(4).values
    out["Grade_(derived)"] = df["Grade"].values
    out["Risk_Tier"] = df["cluster"].values
    out["Risk_Level"] = df["risk_level"].values
    out["Flag_Basis"] = df["Flag basis"].values
    out["In_Intervention_List"] = np.where(df["In intervention list"], "Yes", "No")
    out["Possible_Indicators"] = df["Reasons"].values
    if "actual_class" in df.columns:
        out["Provided_Class_Label"] = df["actual_class"].values
    out.insert(1, "Value_Basis", "Original uploaded values" if basis == "original"
               else "Processed values (winsorized 1st-99th pct, median-imputed)")
    if only_flagged:
        out = out[df["In intervention list"].values].copy()
        out = out.assign(_p=df.loc[df["In intervention list"], "Priority"].values,
                         _s=df.loc[df["In intervention list"], A["score_col"]].values)
        out = out.sort_values(["_p", "_s"]).drop(columns=["_p", "_s"])
    return out


def class_summary(A):
    df, m = A["df"], A["metrics"]
    counts = df["cluster"].value_counts()
    rows = [
        ("Source file", A["name"]), ("Analysis time", A["built"]),
        ("Dataset type", SCHEMAS[A["kind"]]["name"]), ("Students analysed", A["n"]),
        ("Indicators used", len(A["features"])), ("Clusters (K)", m["n_clusters"]),
    ]
    rows += [(f"Students - {t}", int(counts.get(t, 0))) for t in TIERS]
    rows += [(f"Students - Grade {g}", int((df["Grade"] == g).sum())) for g in GRADES]
    rows += [
        ("Students in intervention list (At-Risk tier or Grade F)", int(df["In intervention list"].sum())),
        (f"Mean {A['score_label']} (processed)", round(float(df[A["score_col"]].mean()), 3)),
        ("Silhouette score", round(m["silhouette"], 4)), ("Davies-Bouldin index", round(m["davies_bouldin"], 4)),
        ("Note", "Tiers are relative groupings within this upload, not dropout predictions."),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).astype({"Value": str})


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ----------------------------------------------------------------------------
# Session state, shell
# ----------------------------------------------------------------------------
def init_state():
    for k, v in {"authenticated": False, "page": PAGES[0], "upload": None, "analysis": None}.items():
        st.session_state.setdefault(k, v)


def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def get_analysis():
    return st.session_state.get("analysis")


def require_data(step, title, sub):
    A = get_analysis()
    if A is None:
        page_header(step, title, sub)
        callout("note", "<b>No analysed dataset yet.</b> Upload a CSV and run the analysis first; every number on "
                        "this page is computed from your data.")
        st.button("Go to Upload Dataset", type="primary", on_click=goto, args=(PAGES[0],), key=f"need_{title}")
        return None
    page_header(step, title, sub, A)
    return A


def sidebar():
    A, up = get_analysis(), st.session_state.get("upload")
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-logo">EL</div><div><div class="brand-name">{PRODUCT_NAME}</div>'
            f'<div class="brand-tag">{PRODUCT_TAG}</div></div></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-card"><div class="side-label">Signed in</div>'
            f'<div class="side-value">Welcome, {esc(st.session_state.get("user", "Faculty"))}</div>'
            f'<div class="kpi-sub">{esc(st.session_state.get("role", ""))}</div></div>', unsafe_allow_html=True)
        if A:
            status, color = f"Analysed: {A['name']} ({A['n']} students)", "#10B981"
        elif up:
            status, color = f"Uploaded, not analysed: {up['name']}", "#F59E0B"
        else:
            status, color = "No dataset uploaded", "#94A3B8"
        st.markdown(
            f'<div class="side-card"><div class="side-label">Dataset status</div><div class="side-value">'
            f'<span class="dot" style="background:{color}"></span>{esc(status)}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label" style="margin:.8rem 0 .3rem .2rem">Navigation</div>',
                    unsafe_allow_html=True)
        for i, p in enumerate(PAGES):
            st.button(p, key=f"nav_{i}", use_container_width=True,
                      type="primary" if st.session_state["page"] == p else "secondary",
                      on_click=goto, args=(p,))
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        if st.button("Log out", key="logout_btn", use_container_width=True):
            logout()
            st.rerun()


# ----------------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------------
def login_page():
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            f"""<div class="login-hero">
<div class="brand"><div class="brand-logo">EL</div><div><div class="brand-name">{PRODUCT_NAME}</div>
<div class="brand-tag">{PRODUCT_TAG}</div></div></div>
<div class="login-title">See which students need attention, and why.</div>
<div class="login-desc">Upload a class dataset and get an instant overview of performance, relative risk groups,
and a ready intervention list. Built to support faculty decisions, not to replace them.</div>
<div class="login-point">Clear class overview with grade and risk distributions</div>
<div class="login-point">Intervention list with the indicators behind each flag</div>
<div class="login-point">Cluster analysis explained in plain language, with exportable reports</div>
</div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="login-form-title">Faculty sign in</div>'
                    '<div class="login-form-sub">Enter your details to open the dashboard.</div>',
                    unsafe_allow_html=True)
        with st.form("login_form"):
            role = st.selectbox("Role", ["Faculty Member / Professor", "Course Coordinator", "System Administrator"])
            user = st.text_input("Faculty ID / Username", placeholder="e.g. Prof. Sharma")
            pw = st.text_input("Password", type="password", placeholder="Enter password")
            go_btn = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if go_btn:
            problems = []
            if not user.strip():
                problems.append("Enter your Faculty ID or username.")
            if not pw.strip():
                problems.append("Enter your password.")
            if problems:
                for p in problems:
                    st.error(p)
            else:
                st.session_state.update(authenticated=True, user=user.strip(), role=role, page=PAGES[0])
                st.rerun()
        callout("note", "<b>Demo authentication.</b> This sign-in is a demonstration screen: any username with any "
                        "non-empty password is accepted and nothing is verified or stored. Do not enter a real password.")


# ----------------------------------------------------------------------------
# Page: Upload
# ----------------------------------------------------------------------------
def register_upload(name, data):
    raw, err = read_csv_bytes(data)
    digest = hashlib.md5(data).hexdigest()
    cur = st.session_state.get("upload")
    if cur and cur["digest"] == digest:
        return
    st.session_state["analysis"] = None   # a different file invalidates old results
    if err:
        st.session_state["upload"] = {"name": name, "digest": digest, "raw": None, "error": err, "val": None,
                                      "size": len(data)}
        return
    raw = raw.reset_index(drop=True)
    st.session_state["upload"] = {"name": name, "digest": digest, "raw": raw, "error": None,
                                  "val": validate_upload(raw), "size": len(data)}


def page_upload():
    page_header("Step 1 · Dataset", "Upload & validate your dataset",
                f"Welcome, {esc(st.session_state.get('user', ''))}. Upload a CSV of student records. "
                "The file is checked against what the analysis pipeline actually requires before anything runs.")

    section_title("Supported datasets")
    c1, c2 = st.columns(2)
    with c1:
        card("Academic records (M.Tech style)",
             "<p>Needs at least 3 of these columns: " + ", ".join(esc(c) for c in MTECH_FEATURE_COLS) +
             ".</p><p style='margin-top:.4rem'>Optional: Student ID or Roll Number, " +
             ", ".join(QUIZ_COLS) + ".</p>")
    with c2:
        card("Classroom engagement (xAPI style)",
             "<p>Needs at least 3 of these columns: " + ", ".join(esc(c) for c in XAPI_FEATURE_COLS) +
             ".</p><p style='margin-top:.4rem'>StudentAbsenceDays must be Under-7 / Above-7; ParentAnsweringSurvey "
             "No / Yes; ParentschoolSatisfaction Bad / Good. Optional: Topic, Semester, GradeID, SectionID, "
             "Class (H / M / L).</p>")
    callout("note", f"<b>Record count:</b> {RECORD_RULE} Files with a comma-separated (CSV) format only. "
                    "Sample files tried during development: 57 and 480 records.")

    f = st.file_uploader("Drag and drop a CSV file here, or browse", type=["csv"], key="uploader")
    if f is not None:
        register_upload(f.name, f.getvalue())

    up = st.session_state.get("upload")
    if up is None:
        return
    if up["error"]:
        callout("err", f"<b>Could not read {esc(up['name'])}.</b> {esc(up['error'])}")
        return
    v, raw = up["val"], up["raw"]

    section_title(f"Validation report: {up['name']}")
    if v["errors"]:
        callout("err", "<b>This dataset cannot be analysed.</b><ul>" +
                "".join(f"<li>{esc(e)}</li>" for e in v["errors"]) + "</ul>")
    else:
        callout("ok", f"<b>Dataset passed the pipeline's requirements</b> and was recognised as "
                      f"{esc(SCHEMAS[v['kind']]['name'])}.")
    if v["warnings"]:
        callout("warn", "<b>Please review:</b><ul>" + "".join(f"<li>{esc(w)}</li>" for w in v["warnings"]) + "</ul>")

    cells = v["n"] * max(len(v["used"]), 1)
    kpis([
        ("Records", f"{v['n']:,}", "student rows", PRIMARY),
        ("Columns in file", v["n_cols"], "total", "#64748B"),
        ("Indicators used", len(v["used"]), "recognised by pipeline", PRIMARY_DARK),
        ("Missing values", f"{v['missing_cells']:,}", f"{pct(v['missing_cells'], cells)} of indicator cells", "#F59E0B"),
        ("Duplicate rows", v["dup_rows"], "exact duplicates", "#F59E0B" if v["dup_rows"] else "#10B981"),
    ])
    t1, t2 = st.tabs(["Column check", "Data preview (as uploaded)"])
    with t1:
        if v["rows"]:
            show_df(pd.DataFrame(v["rows"]), height=420)
            st.caption("Missing = empty cells. Invalid = values that cannot be read as the expected type. "
                       "Both are filled with the column median during processing.")
    with t2:
        show_df(raw.head(10))
        st.caption(f"First {min(10, len(raw))} of {len(raw):,} rows, exactly as uploaded.")

    A = get_analysis()
    if A and A["name"] == up["name"]:
        callout("ok", f"<b>Analysis complete</b> for {A['n']} students.")
        st.button("Open Class Overview", type="primary", on_click=goto, args=("Class Overview",), key="to_overview")
    elif not v["errors"]:
        if st.button("Run analysis", type="primary", key="run_analysis"):
            try:
                with st.spinner("Cleaning data, clustering students and preparing reports..."):
                    st.session_state["analysis"] = build_analysis(up["name"], raw)
                st.rerun()
            except Exception as exc:
                callout("err", f"<b>Analysis failed.</b> {esc(exc)}<br>Check the column check table above for "
                               "columns with no valid values.")


# ----------------------------------------------------------------------------
# Chart builders
# ----------------------------------------------------------------------------
def fig_grade_bar(df, title="Grade distribution"):
    c = df["Grade"].value_counts()
    order = [g for g in GRADES + ["Unknown"] if g in c.index or g in GRADES]
    d = pd.DataFrame({"Grade": order, "Students": [int(c.get(g, 0)) for g in order]})
    d["Share"] = (d["Students"] / max(len(df), 1) * 100).round(1)
    fig = px.bar(d, x="Grade", y="Students", color="Grade", color_discrete_map=GRADE_COLORS, text="Students",
                 custom_data=["Share"])
    fig.update_traces(hovertemplate="Grade %{x}<br>%{y} students (%{customdata[0]}%)<extra></extra>",
                      textposition="outside")
    return style_fig(fig, 340, title)


def fig_tier_donut(df, title="Risk tier distribution"):
    c = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int)
    if c.sum() == 0:
        return None
    fig = go.Figure(go.Pie(labels=list(c.index), values=list(c.values), hole=0.55, sort=False,
                           marker=dict(colors=[CLUSTER_COLORS[t] for t in c.index]),
                           textinfo="label+percent", hovertemplate="%{label}<br>%{value} students<extra></extra>"))
    return style_fig(fig, 340, title)


def fig_score_hist(A, color="cluster", title=None):
    df, sc = A["df"], A["score_col"]
    if not has_values(df, sc):
        return None
    fig = px.histogram(df, x=sc, color=color, nbins=20, barmode="overlay", opacity=0.75,
                       color_discrete_map={**CLUSTER_COLORS, **GRADE_COLORS},
                       category_orders={"cluster": TIERS, "Grade": GRADES},
                       labels={sc: A["score_label"], "cluster": "Tier"})
    fig.update_layout(yaxis_title="Students", xaxis_title=A["score_label"])
    fig.add_vline(x=float(df[sc].mean()), line_dash="dash", line_color=NAVY,
                  annotation_text=f"Mean {df[sc].mean():.1f}", annotation_position="top")
    return style_fig(fig, 340, title or f"{A['score_label']} distribution", legend=True)


def fig_box(df, y, ylabel, by="cluster", title=None, order=TIERS, cmap=None):
    if not has_values(df, y):
        return None
    fig = px.box(df, x=by, y=y, color=by, points="all", color_discrete_map=cmap or CLUSTER_COLORS,
                 category_orders={by: order}, hover_data=[c for c in ["Student ID"] if c in df.columns])
    fig.update_layout(xaxis_title="Risk tier" if by == "cluster" else by, yaxis_title=ylabel)
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    return style_fig(fig, 340, title or f"{ylabel} by {'risk tier' if by == 'cluster' else by}")


def grade_caveat(A):
    if A["kind"] == "xapi":
        callout("warn", "<b>How Grade is derived for this dataset.</b> The file holds engagement counts, not exam marks. "
                        f"Grade is a band of the average engagement count ({GRADE_RULE}), so Grade F here means low "
                        "recorded engagement relative to those fixed cut-offs and can cover a large share of the "
                        "class. Treat it as an engagement band, not an academic grade.")
    else:
        callout("note", f"<b>Grade bands</b> are derived from the AVG column: {GRADE_RULE}. Grade and risk tier are "
                        "separate labels: a student can be Grade F and still sit in the Moderate tier.")


# ----------------------------------------------------------------------------
# Page: Class overview
# ----------------------------------------------------------------------------
def page_overview():
    A = require_data("Step 2 · Overview", "Class overview",
                     "A snapshot of the uploaded class: how many students sit in each risk tier and grade band.")
    if A is None:
        return
    df, n = A["df"], A["n"]
    cnt = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int)
    f_cnt = int((df["Grade"] == "F").sum())
    flagged = int(df["In intervention list"].sum())
    kpis([
        ("Total students", f"{n:,}", "in uploaded file", PRIMARY),
        ("At-Risk tier", int(cnt["At-Risk"]), pct(cnt["At-Risk"], n) + " of class", CLUSTER_COLORS["At-Risk"]),
        ("Moderate tier", int(cnt["Moderate"]), pct(cnt["Moderate"], n) + " of class", CLUSTER_COLORS["Moderate"]),
        ("High-Performing tier", int(cnt["High-Performing"]), pct(cnt["High-Performing"], n) + " of class",
         CLUSTER_COLORS["High-Performing"]),
        (f"Mean {A['score_label']}", f"{df[A['score_col']].mean():.1f}", "processed values", PRIMARY_DARK),
        ("Grade F", f_cnt, pct(f_cnt, n) + " of class", "#DC2626"),
        ("Intervention list", flagged, "At-Risk tier or Grade F", NAVY2),
    ])
    callout("info", "<b>Reading this page.</b> Tiers compare students <i>within this upload</i>. They are decision "
                    "support for faculty, not predictions of dropout or final results.")
    grade_caveat(A)
    a, b = st.columns(2)
    with a:
        plot(fig_grade_bar(df), "ov_grade")
    with b:
        plot(fig_tier_donut(df), "ov_tier")
    a, b = st.columns(2)
    with a:
        plot(fig_score_hist(A), "ov_hist")
    with b:
        plot(fig_box(df, A["score_col"], A["score_label"]), "ov_box")

    section_title("Class statistics (processed values)")
    stats = df[A["features"] + [A["score_col"]]].describe().T[["mean", "50%", "std", "min", "max"]].round(2)
    stats.index = [label(c) if c != A["score_col"] else A["score_label"] + " (derived)" for c in stats.index]
    stats.columns = ["Mean", "Median", "Std. dev.", "Min", "Max"]
    show_df(stats.reset_index().rename(columns={"index": "Indicator"}), height=min(460, 60 + 36 * len(stats)))
    st.caption("Processed values: extreme values clipped to the 1st-99th percentile and missing values filled with "
               "the median (see Indicator Summary).")
    st.button("Review students needing attention", type="primary", on_click=goto, args=("At-Risk Students",),
              key="ov_to_risk")


# ----------------------------------------------------------------------------
# Page: Student performance
# ----------------------------------------------------------------------------
def page_performance():
    A = require_data("Step 3 · Performance", "Student performance analysis",
                     "How grades, scores and engagement indicators are distributed across the class and the risk tiers.")
    if A is None:
        return
    df, sch = A["df"], SCHEMAS[A["kind"]]
    grade_caveat(A)
    tabs = ["Grades & scores", "Key indicators"]
    if A["quiz_cols"]:
        tabs.append("Quiz performance")
    groupers = [c for c in ["Topic", "Semester", "GradeID", "SectionID", "StageID"]
                if c in df.columns and 1 < df[c].nunique() <= 20]
    if groupers:
        tabs.append("Course comparison")
    tab = dict(zip(tabs, st.tabs(tabs)))

    with tab["Grades & scores"]:
        a, b = st.columns(2)
        with a:
            m = df.groupby(["cluster", "Grade"]).size().reset_index(name="Students")
            fig = px.bar(m, x="cluster", y="Students", color="Grade", barmode="stack",
                         color_discrete_map=GRADE_COLORS, category_orders={"cluster": TIERS, "Grade": GRADES})
            fig.update_layout(xaxis_title="Risk tier")
            plot(style_fig(fig, 340, "Grades within each risk tier", legend=True), "pf_gt")
        with b:
            plot(fig_box(df, A["score_col"], A["score_label"], by="Grade", order=GRADES, cmap=GRADE_COLORS,
                         title=f"{A['score_label']} by grade"), "pf_sg")
        plot(fig_score_hist(A, color="Grade"), "pf_hist")

    with tab["Key indicators"]:
        keys = [c for c in sch["key"] if has_values(df, c)]
        cols = st.columns(2)
        for i, c in enumerate(keys):
            with cols[i % 2]:
                plot(fig_box(df, c, label(c)), f"pf_key_{c}")
        if A["kind"] == "mtech":
            st.caption("The pipeline does not include an assignment-marks column; quiz results are shown in the Quiz "
                       "performance tab.")
        section_title("Explore any indicator")
        ind = st.selectbox("Indicator", A["features"], format_func=label, key="pf_ind")
        a, b = st.columns(2)
        with a:
            if has_values(df, ind):
                fig = px.histogram(df, x=ind, color="cluster", nbins=20, barmode="overlay", opacity=0.75,
                                   color_discrete_map=CLUSTER_COLORS, category_orders={"cluster": TIERS},
                                   labels={ind: label(ind), "cluster": "Tier"})
                fig.update_layout(yaxis_title="Students")
                plot(style_fig(fig, 340, f"{label(ind)} distribution", legend=True), "pf_ind_h")
            else:
                plot(None)
        with b:
            plot(fig_box(df, ind, label(ind)), "pf_ind_b")

    if A["quiz_cols"]:
        with tab["Quiz performance"]:
            q = df.groupby("cluster")[A["quiz_cols"]].mean().reindex(TIERS)
            long = q.reset_index().melt(id_vars="cluster", var_name="Quiz", value_name="Mean score")
            long = long.dropna()
            fig = px.line(long, x="Quiz", y="Mean score", color="cluster", markers=True,
                          color_discrete_map=CLUSTER_COLORS, category_orders={"cluster": TIERS},
                          labels={"cluster": "Tier"}) if len(long) else None
            plot(style_fig(fig, 360, "Mean quiz score by risk tier", legend=True) if fig else None, "pf_quiz")
            st.caption("Quiz columns are shown exactly as uploaded (the pipeline does not alter them). Empty quiz "
                       "cells are left out of the averages.")

    if groupers:
        with tab["Course comparison"]:
            g = st.selectbox("Compare by", groupers, key="pf_group")
            d = df.groupby(g).agg(Students=(A["id_col"], "count"), Mean=(A["score_col"], "mean"),
                                  AtRisk=("cluster", lambda s: (s == "At-Risk").mean() * 100)).reset_index()
            a, b = st.columns(2)
            with a:
                fig = px.bar(d, x=g, y="Mean", text=d["Mean"].round(1), color_discrete_sequence=[PRIMARY],
                             hover_data={"Students": True})
                fig.update_layout(yaxis_title=A["score_label"])
                plot(style_fig(fig, 340, f"Mean {A['score_label']} by {g}"), "pf_cg1")
            with b:
                fig = px.bar(d, x=g, y="AtRisk", text=d["AtRisk"].round(1), color_discrete_sequence=[CLUSTER_COLORS["At-Risk"]],
                             hover_data={"Students": True})
                fig.update_layout(yaxis_title="% of students in At-Risk tier")
                plot(style_fig(fig, 340, f"At-Risk share by {g}"), "pf_cg2")
            st.caption("Group sizes can be small; check the Students count in the tooltip before drawing conclusions.")


# ----------------------------------------------------------------------------
# Page: At-risk students
# ----------------------------------------------------------------------------
def style_tier(styler, cols):
    colors = {"At-Risk": "color:#991B1B;font-weight:700", "Moderate": "color:#92400E;font-weight:600",
              "High-Performing": "color:#065F46;font-weight:600"}
    fn = lambda v: colors.get(v, "")
    return (styler.map(fn, subset=cols) if hasattr(styler, "map") else styler.applymap(fn, subset=cols))


def page_risk():
    A = require_data("Step 4 · Intervention", "At-risk student identification",
                     "Students who may benefit from a conversation or extra support, with the indicators behind each flag.")
    if A is None:
        return
    df, sch, idc = A["df"], SCHEMAS[A["kind"]], A["id_col"]
    fl = df[df["In intervention list"]]
    both = int((fl["Flag basis"] == "At-Risk tier + Grade F").sum())
    f_only = int((fl["Flag basis"] == "Grade F only").sum())
    f_mod = int(((fl["Grade"] == "F") & (fl["cluster"] == "Moderate")).sum())
    kpis([
        ("Students flagged", len(fl), pct(len(fl), A["n"]) + " of class", NAVY2),
        ("At-Risk tier", int((df["cluster"] == "At-Risk").sum()), "cluster label", CLUSTER_COLORS["At-Risk"]),
        ("Grade F", int((df["Grade"] == "F").sum()), "derived grade", "#DC2626"),
        ("Both flags", both, "highest priority", PRIMARY_DARK),
        ("Grade F, not At-Risk tier", f_only, f"{f_mod} are in Moderate tier", "#F59E0B"),
    ])
    callout("info", "<b>Who is listed:</b> every student in the At-Risk tier <i>or</i> with Grade F. A Grade F student "
                    "is never left out because their cluster is Moderate.")
    grade_caveat(A)
    callout("warn", "<b>Please read:</b> these indicators support faculty decisions and do not predict individual "
                    "outcomes. 'Possible indicators' only point out values that are low compared with the rest of this "
                    "class (below its lower quartile). A low value is an association in this data, not a proven cause. "
                    "The derived score itself is not listed as a reason, because Grade is calculated from it.")
    if A["kind"] == "xapi":
        st.caption("This dataset has no attendance or quiz-attempt columns. The closest available indicators "
                   "(participation, resource use, absence days) are shown.")
    if fl.empty:
        callout("ok", "No students are currently flagged.")
        return

    f1, f2, f3, f4 = st.columns([1.3, 1.2, 1, 1.2])
    basis = f1.multiselect("Flag basis", ["At-Risk tier + Grade F", "At-Risk tier only", "Grade F only"],
                           default=["At-Risk tier + Grade F", "At-Risk tier only", "Grade F only"], key="rk_basis")
    tiers = f2.multiselect("Risk tier", TIERS, default=TIERS, key="rk_tier")
    grades = f3.multiselect("Grade", GRADES, default=GRADES, key="rk_grade")
    search = f4.text_input("Search student ID", key="rk_search", placeholder="Type part of an ID")

    cols_ind = [c for c in sch["table"] if c in df.columns]
    sort_opts = ["Priority (default)", "Derived score"] + [label(c) for c in cols_ind]
    s1, s2, s3 = st.columns([1.3, 1, 2.2])
    sort_by = s1.selectbox("Sort by", sort_opts, key="rk_sort")
    asc = s2.radio("Order", ["Ascending", "Descending"], horizontal=True, key="rk_order") == "Ascending"
    only_reasons = s3.checkbox("Only students with at least one indicator below the class lower quartile",
                               key="rk_reason")

    view = fl[fl["Flag basis"].isin(basis) & fl["cluster"].isin(tiers) & fl["Grade"].isin(grades)]
    if search.strip():
        view = view[view[idc].astype(str).str.contains(search.strip(), case=False, na=False, regex=False)]
    if only_reasons:
        view = view[view["_n_reasons"] > 0]
    if sort_by.startswith("Priority"):
        view = view.sort_values(["Priority", A["score_col"]], ascending=[True, True])
        if not asc:
            view = view.iloc[::-1]
    else:
        key = A["score_col"] if sort_by == "Derived score" else next(c for c in cols_ind if label(c) == sort_by)
        view = view.sort_values(key, ascending=asc)

    show = pd.DataFrame({"Student ID": view[idc].values, "Flag basis": view["Flag basis"].values,
                         "Risk tier": view["cluster"].values, "Risk level": view["risk_level"].values,
                         "Grade": view["Grade"].values, A["score_label"]: view[A["score_col"]].round(1).values})
    for c in cols_ind:
        show[label(c)] = view[c].round(1).values
    show["Possible indicators"] = view["Reasons"].values
    st.markdown(f"**{len(view)}** of {len(fl)} flagged students shown", unsafe_allow_html=False)
    if show.empty:
        callout("note", "No students match the current filters.")
    else:
        st.dataframe(style_tier(show.style, ["Risk tier"]), use_container_width=True, hide_index=True, height=480)
    st.caption("Values are processed values (outliers clipped to the 1st-99th percentile, missing values filled with "
               "the median). Original uploaded values appear on the Student Details page.")

    with st.container(border=True):
        c = view["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int) if len(view) else None
        if c is None or c.sum() == 0:
            st.markdown('<div class="empty">No students to chart for the current filters.</div>', unsafe_allow_html=True)
        else:
            gcount = view.groupby(["cluster", "Grade"]).size().reset_index(name="Students")
            fig = px.bar(gcount, x="cluster", y="Students", color="Grade", color_discrete_map=GRADE_COLORS,
                         category_orders={"cluster": TIERS, "Grade": GRADES}, barmode="stack")
            fig.update_layout(xaxis_title="Risk tier")
            st.plotly_chart(style_fig(fig, 300, "Filtered students by tier and grade", legend=True),
                            use_container_width=True, key="rk_chart")


# ----------------------------------------------------------------------------
# Page: Cluster analysis
# ----------------------------------------------------------------------------
def sil_note(s):
    if s < 0.25:
        return "Little clear structure: the groups overlap a lot."
    if s < 0.5:
        return "Weak to moderate separation: groups overlap, so treat tiers as a rough guide."
    return "Clearer separation between groups."


def db_note(d):
    return ("Lower is better. Above about 1, groups are wider than the gap between them."
            if d > 1 else "Lower is better. Below 1 suggests groups are fairly compact relative to their gaps.")


def metric_card(title, value, note):
    st.markdown(f'<div class="card" style="border-top:3px solid #F59E0B"><div class="side-label">{esc(title)}</div>'
                f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub" style="margin-top:.5rem">{esc(note)}</div>'
                '</div>', unsafe_allow_html=True)


def page_cluster():
    A = require_data("Step 5 · Clusters", "Cluster analysis",
                     "How the pipeline grouped students, what distinguishes the groups, and how clear the grouping is.")
    if A is None:
        return
    df, m = A["df"], A["metrics"]
    extra = ""
    if A["kind"] == "xapi":
        extra = (" All indicators are treated as 'higher = stronger' in this ranking, including the absence-over-7-days "
                 "indicator, because the pipeline averages the normalised values without reversing any column.")
    card("How groups and tier names are assigned",
         "<ul><li><b>Step 1:</b> each indicator is cleaned (extreme values clipped, gaps filled with the median) and "
         f"standardised.</li><li><b>Step 2:</b> K-Means splits students into {m['n_clusters']} groups of similar "
         "profiles.</li><li><b>Step 3:</b> every indicator is rescaled to 0-1 across the class, and each group gets the "
         "average of those values for its members. The group with the highest average is named High-Performing, the "
         "lowest At-Risk, and the remaining one Moderate.</li></ul>"
         "<p style='margin-top:.5rem'>So tiers are relative to this class and are not fixed score bands, quartiles or "
         f"dropout forecasts.{esc(extra)}</p>")

    sil, dbi = m["silhouette"], m["davies_bouldin"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Silhouette score", f"{sil:.3f}", sil_note(sil) + " Ranges from -1 to 1; higher means clearer groups.")
    with c2:
        metric_card("Davies-Bouldin index", f"{dbi:.3f}", db_note(dbi))
    with c3:
        metric_card("Clusters (K)", str(m["n_clusters"]), "Fixed by the pipeline (N_CLUSTERS).")
    with c4:
        metric_card("PCA variance (2D map)", f"{A['pca_variance'].sum() * 100:.1f}%",
                    "Share of variation the 2D map below keeps. The rest is not visible in the plot.")
    callout("warn", "These measures describe how well separated the groups are, not how accurate predictions are. "
                    "Overlapping groups mean some students near a boundary could reasonably belong to the neighbouring tier.")

    a, b = st.columns(2)
    with a:
        c = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int).reset_index()
        c.columns = ["Tier", "Students"]
        fig = px.bar(c, x="Tier", y="Students", color="Tier", color_discrete_map=CLUSTER_COLORS, text="Students")
        fig.update_traces(textposition="outside")
        plot(style_fig(fig, 340, "Cluster distribution"), "cl_dist")
    with b:
        lab2tier = df.drop_duplicates("kmeans_label").set_index("kmeans_label")["cluster"].to_dict()
        sc = pd.DataFrame([(lab2tier.get(int(k), str(k)), v) for k, v in m["cluster_performance_scores"].items()],
                          columns=["Tier", "Average normalised score"]).set_index("Tier").reindex(TIERS).reset_index()
        fig = px.bar(sc, x="Tier", y="Average normalised score", color="Tier", color_discrete_map=CLUSTER_COLORS,
                     text=sc["Average normalised score"].round(3))
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0, 1])
        plot(style_fig(fig, 340, "Ranking score used to name the tiers (0-1)"), "cl_rank")

    a, b = st.columns(2)
    with a:
        hover = [c for c in [A["id_col"], A["score_col"], "Grade"] if c in df.columns]
        fig = px.scatter(df, x="pca_x", y="pca_y", color="cluster", color_discrete_map=CLUSTER_COLORS,
                         category_orders={"cluster": TIERS}, hover_data=hover,
                         labels={"pca_x": "Principal component 1", "pca_y": "Principal component 2", "cluster": "Tier"})
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        plot(style_fig(fig, 400, "Students on a 2D map (PCA)", legend=True), "cl_pca")
    with b:
        mm = df[A["features"]].copy()
        rng = (mm.max() - mm.min()).replace(0, np.nan)
        norm = ((mm - mm.min()) / rng).fillna(0)
        heat = norm.groupby(df["cluster"]).mean().reindex(TIERS)
        heat.columns = [label(c) for c in heat.columns]
        fig = px.imshow(heat.values, x=list(heat.columns), y=list(heat.index), text_auto=".2f", aspect="auto",
                        color_continuous_scale="Purples", zmin=0, zmax=1)
        fig.update_layout(coloraxis_colorbar=dict(title="0-1"))
        fig.update_xaxes(tickangle=-40)
        plot(style_fig(fig, 400, "Cluster comparison (average normalised value)"), "cl_heat")

    section_title("Cluster characteristics (average processed values)")
    prof = df.groupby("cluster")[A["features"]].mean().reindex(TIERS).T
    prof.index = [label(c) for c in prof.index]
    prof["Class mean"] = df[A["features"]].mean().values
    show_df(prof.round(2).reset_index().rename(columns={"index": "Indicator"}), height=min(470, 60 + 36 * len(prof)))
    sizes = df.groupby("cluster").agg(Students=(A["id_col"], "count"),
                                      GradeF=("Grade", lambda s: int((s == "F").sum()))).reindex(TIERS).reset_index()
    sizes.columns = ["Tier", "Students", "Grade F students"]
    show_df(sizes)
    if "ari_vs_actual_class" in m:
        st.caption(f"Agreement with the file's own Class label (H/M/L): adjusted Rand index "
                   f"{m['ari_vs_actual_class']:.3f}, normalised mutual information {m['nmi_vs_actual_class']:.3f}. "
                   "Values near 0 mean little agreement beyond chance; this is agreement, not accuracy.")


# ----------------------------------------------------------------------------
# Page: Model evaluation
# ----------------------------------------------------------------------------
def page_model():
    A = require_data("Step 6 · Evaluation", "Model evaluation",
                     "Checks on the grouping model. K-Means is unsupervised, so there is no accuracy score or training "
                     "curve: these are cluster-quality measures.")
    if A is None:
        return
    df, m = A["df"], A["metrics"]
    t1, t2, t3 = st.tabs(["Choice of K", "Score distribution" + (" & quiz trend" if A["quiz_cols"] else ""),
                          "Agreement with provided labels"])
    with t1:
        callout("note", f"The pipeline uses K = {m['n_clusters']}. The table below re-runs the backend's own "
                        "evaluate_k_values to show how other values of K compare on this data. It does not change "
                        "the analysis.")
        if A["k_eval"] is None:
            with st.spinner("Comparing different numbers of clusters..."):
                try:
                    A["k_eval"] = evaluate_k_values(A["X_scaled"])
                except Exception as exc:
                    A["k_eval"] = pd.DataFrame()
                    st.error(f"Could not compute the comparison: {exc}")
        k = A["k_eval"]
        if k is None or k.empty:
            plot(None)
        else:
            cols = st.columns(3)
            specs = [("Inertia", "Inertia (lower = tighter groups; always falls as K grows)"),
                     ("Silhouette", "Silhouette (higher = clearer separation)"),
                     ("Davies-Bouldin", "Davies-Bouldin (lower = clearer separation)")]
            for col, (name, title) in zip(cols, specs):
                with col:
                    fig = px.line(k, x="K", y=name, markers=True, color_discrete_sequence=[PRIMARY])
                    fig.add_vline(x=m["n_clusters"], line_dash="dash", line_color="#F59E0B",
                                  annotation_text="K used", annotation_position="top")
                    fig.update_xaxes(dtick=1)
                    plot(style_fig(fig, 320, title), f"me_{name}")
            show_df(k.round(4))
            st.caption("Different measures can disagree. Use them as a sanity check, not as proof that a particular K is correct.")
    with t2:
        a, b = st.columns(2)
        with a:
            plot(fig_score_hist(A), "me_hist")
        with b:
            if A["quiz_cols"]:
                q = df.groupby("cluster")[A["quiz_cols"]].mean().reindex(TIERS).reset_index().melt(
                    id_vars="cluster", var_name="Quiz", value_name="Mean score").dropna()
                fig = px.line(q, x="Quiz", y="Mean score", color="cluster", markers=True,
                              color_discrete_map=CLUSTER_COLORS, category_orders={"cluster": TIERS},
                              labels={"cluster": "Tier"}) if len(q) else None
                plot(style_fig(fig, 340, "Mean quiz score by tier", legend=True) if fig else None, "me_quiz")
            else:
                prof = df.groupby("cluster")[A["features"]].mean().reindex(TIERS).T
                prof.index = [label(c) for c in prof.index]
                long = prof.reset_index().melt(id_vars="index", var_name="Tier", value_name="Mean")
                fig = px.bar(long, x="index", y="Mean", color="Tier", barmode="group",
                             color_discrete_map=CLUSTER_COLORS, labels={"index": "Indicator"})
                fig.update_xaxes(tickangle=-35)
                plot(style_fig(fig, 340, "Average indicator by tier", legend=True), "me_prof")
    with t3:
        if "actual_class" in df.columns and df["actual_class"].notna().any():
            ct = pd.crosstab(df["cluster"], df["actual_class"]).reindex(TIERS)
            ct = ct[[c for c in ["High", "Medium", "Low"] if c in ct.columns]]
            c1, c2 = st.columns(2)
            with c1:
                metric_card("Adjusted Rand index", f"{m.get('ari_vs_actual_class', float('nan')):.3f}",
                            "0 is chance-level agreement, 1 is identical grouping.")
            with c2:
                metric_card("Normalised mutual information", f"{m.get('nmi_vs_actual_class', float('nan')):.3f}",
                            "0 means no shared information, 1 means identical grouping.")
            fig = px.imshow(ct.values, x=list(ct.columns), y=list(ct.index), text_auto=True, aspect="auto",
                            color_continuous_scale="Purples")
            fig.update_layout(xaxis_title="Class label provided in the file", yaxis_title="Tier assigned by clustering")
            plot(style_fig(fig, 340, "Provided class label vs assigned tier (student counts)"), "me_ct")
            st.caption("The file's Class label is not used to build the clusters. Agreement measures how similar the two "
                       "groupings are; it is not a prediction accuracy.")
        else:
            callout("note", "This dataset has no recognised class-label column (H / M / L), so there is nothing to compare against.")


# ----------------------------------------------------------------------------
# Page: Indicator summary
# ----------------------------------------------------------------------------
def page_indicators():
    A = require_data("Step 7 · Indicators", "Indicator summary",
                     "Descriptive comparison of the available indicators across risk tiers or grades.")
    if A is None:
        return
    df = A["df"]
    callout("warn", "These are descriptive summaries of columns in your file. They are <b>not</b> formally measured "
                    "Course Outcome attainment values.")
    group = st.radio("Compare across", ["Risk tier", "Grade"], horizontal=True, key="is_group")
    by, order, cmap = ("cluster", TIERS, CLUSTER_COLORS) if group == "Risk tier" else ("Grade", GRADES, GRADE_COLORS)
    present = [o for o in order if (df[by] == o).any()]

    means = df.groupby(by)[A["features"]].mean().reindex(present)
    long = means.T.copy()
    long.index = [label(c) for c in long.index]
    a, b = st.columns([3, 2])
    with a:
        fig = px.bar(long.reset_index().melt(id_vars="index", var_name=group, value_name="Mean"),
                     x="index", y="Mean", color=group, barmode="group", color_discrete_map=cmap,
                     labels={"index": "Indicator"}, category_orders={group: present})
        fig.update_xaxes(tickangle=-35)
        plot(style_fig(fig, 400, f"Average of each indicator by {group.lower()} (processed values)", legend=True), "is_bar")
    with b:
        mm = df[A["features"]]
        rng = (mm.max() - mm.min()).replace(0, np.nan)
        norm = ((mm - mm.min()) / rng).fillna(0).groupby(df[by]).mean().reindex(present)
        norm.columns = [label(c) for c in norm.columns]
        fig = px.imshow(norm.T.values, x=list(norm.index), y=list(norm.columns), text_auto=".2f", aspect="auto",
                        color_continuous_scale="Purples", zmin=0, zmax=1)
        plot(style_fig(fig, 400, "Normalised comparison (0-1, for comparison only)"), "is_heat")

    section_title("How each indicator was prepared")
    sch = SCHEMAS[A["kind"]]
    rows = []
    for t in A["trail"]:
        c = t["col"]
        rows.append({
            "Indicator": label(c),
            "Raw (as uploaded)": "Text converted to 0/1 by the pipeline" if (A["kind"] == "xapi" and c in XAPI_BINARY_MAP) else "Numeric",
            "Missing values filled (median)": t["imputed"],
            "Values clipped (1st-99th pct)": t["clipped"],
            "Clip range": f"{fmt(t['lo'])} to {fmt(t['hi'])}",
            "Unusual values (IQR rule)": t["outliers"],
        })
    show_df(pd.DataFrame(rows), height=min(480, 60 + 36 * len(rows)))
    st.markdown(
        "<div class='card'><h4>Value types used in this dashboard</h4><ul>"
        "<li><b>Raw:</b> exactly as uploaded. Shown on Student Details and in the 'original' export.</li>"
        "<li><b>Processed:</b> raw values after clipping to the 1st-99th percentile and median imputation. Used in "
        "most charts and tables.</li>"
        "<li><b>Standardised:</b> processed values rescaled for K-Means. Internal only; never displayed as scores.</li>"
        "<li><b>Normalised 0-1:</b> processed values rescaled across the class. Used for the tier ranking and the "
        "comparison heatmaps.</li>"
        f"<li><b>Derived:</b> the score and Grade. {esc(sch['score_note'])}</li></ul></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Page: Student details
# ----------------------------------------------------------------------------
def page_student():
    A = require_data("Step 8 · Student", "Student details",
                     "Inspect one student's indicators alongside the class, with considerations for follow-up.")
    if A is None:
        return
    df, orig, idc = A["df"], A["orig"], A["id_col"]
    dup_ids = df[idc].duplicated(keep=False)
    options = list(range(len(df)))
    fmt_fn = lambda i: f"{df[idc].iat[i]}" + (f" (row {i + 1})" if dup_ids.iat[i] else "")
    i = st.selectbox("Select a student (type to search)", options, format_func=fmt_fn, key="sd_pick")
    r = df.iloc[i]

    kpis([
        ("Student ID", str(r[idc]), "", NAVY2),
        ("Grade", r["Grade"], "derived from score", GRADE_COLORS.get(r["Grade"], MUTED)),
        ("Risk tier", r["cluster"], f"risk level: {r['risk_level']}", CLUSTER_COLORS.get(r["cluster"], MUTED)),
        (A["score_label"], f"{r[A['score_col']]:.1f}", f"class mean {df[A['score_col']].mean():.1f}", PRIMARY),
        ("Intervention list", "Yes" if r["In intervention list"] else "No", r["Flag basis"] or "not flagged",
         "#DC2626" if r["In intervention list"] else "#10B981"),
    ])

    rows, pcts = [], []
    for c in A["features"]:
        q1, med = A["quartiles"][c], df[c].median()
        q3 = df[c].quantile(0.75)
        v = r[c]
        pos = "Below class lower quartile" if v < q1 else ("Above class upper quartile" if v > q3 else "Within middle half of class")
        if A["kind"] == "xapi" and c in XAPI_BINARY_MAP:
            pos = "-"
        rows.append({"Indicator": label(c), "Original (as uploaded)": str(orig[c].iat[i]),
                     "Processed value": fmt(v), "Class median": fmt(med), "Position in class": pos})
        if df[c].nunique() > 2:
            pcts.append((label(c), float(df[c].rank(pct=True).iat[i] * 100)))

    a, b = st.columns([3, 2])
    with a:
        section_title("Indicators: original and processed values")
        show_df(pd.DataFrame(rows), height=min(520, 60 + 36 * len(rows)))
        st.caption("'Original' is the value in your file. 'Processed' is after clipping and median filling, as used in the "
                   "analysis. They differ only where a value was missing or extreme.")
    with b:
        if pcts:
            d = pd.DataFrame(pcts, columns=["Indicator", "Percentile"])
            fig = px.bar(d, x="Percentile", y="Indicator", orientation="h", range_x=[0, 100],
                         color_discrete_sequence=[PRIMARY], text=d["Percentile"].round(0))
            fig.add_vline(x=50, line_dash="dash", line_color=MUTED, annotation_text="Class median")
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Percentile within this class")
            plot(style_fig(fig, 420, "Where this student sits in the class"), "sd_pct")
        else:
            plot(None)

    section_title("Possible intervention considerations")
    rc = r["_reason_cols"]
    if rc:
        items = []
        for c in rc:
            text = CONSIDERATIONS.get(c, CONSIDERATIONS["_engagement"])
            items.append(f"<li><b>{esc(label(c))}:</b> {esc(text)}</li>")
        body = "<p>Indicators below the class lower quartile (or absence flag):</p><ul>" + "".join(items) + "</ul>"
    elif r["In intervention list"]:
        body = ("<p>This student is flagged by "
                f"{esc(r['Flag basis'])}, but no single indicator is below the class lower quartile. "
                "Review the full record before deciding on any action.</p>")
    else:
        body = "<p>This student is not on the intervention list and no indicator is below the class lower quartile.</p>"
    card("For faculty consideration", body)
    callout("note", "These are prompts based on this student's recorded indicators relative to the class. They do not "
                    "explain why a value is low and do not predict outcomes; use your own knowledge of the student.")


# ----------------------------------------------------------------------------
# Page: Export
# ----------------------------------------------------------------------------
def page_export():
    A = require_data("Step 9 · Export", "Export reports",
                     "Download results as CSV files you can open in Excel or share with colleagues.")
    if A is None:
        return
    callout("info", "<b>Choose which values to export.</b> Original values are exactly as uploaded. Processed values were "
                    "clipped to the 1st-99th percentile and had missing entries filled with the median. Every file "
                    "includes a Value_Basis column stating which one it holds. Derived score, Grade and tiers are always "
                    "calculated from processed values.")
    choice = st.radio("Values for student columns",
                      ["Original uploaded values", "Processed values (used for analysis)"], horizontal=True, key="ex_basis")
    basis = "original" if choice.startswith("Original") else "processed"
    stamp = datetime.now().strftime("%Y%m%d")

    res = results_table(A, basis)
    flagged = results_table(A, basis, only_flagged=True)
    summ = class_summary(A)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Student results", f"<p>{len(res):,} students · {res.shape[1]} columns</p>")
        st.download_button("Download student results (CSV)", to_csv_bytes(res),
                           f"student_results_{basis}_{stamp}.csv", "text/csv", use_container_width=True, key="dl_res")
    with c2:
        card("Intervention list", f"<p>{len(flagged):,} students (At-Risk tier or Grade F)</p>")
        st.download_button("Download intervention list (CSV)", to_csv_bytes(flagged),
                           f"intervention_list_{basis}_{stamp}.csv", "text/csv", use_container_width=True,
                           key="dl_flag", disabled=flagged.empty)
    with c3:
        card("Class summary", f"<p>{len(summ)} summary rows · counts and quality metrics</p>")
        st.download_button("Download class summary (CSV)", to_csv_bytes(summ), f"class_summary_{stamp}.csv",
                           "text/csv", use_container_width=True, key="dl_sum")
    section_title("Preview of student results")
    show_df(res.head(15))
    st.caption("Quiz columns (if present) are never altered by the pipeline and are identical in both versions. "
               "These files support faculty decisions and do not predict outcomes.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
ROUTES = {
    "Upload Dataset": page_upload, "Class Overview": page_overview, "Student Performance": page_performance,
    "At-Risk Students": page_risk, "Cluster Analysis": page_cluster, "Model Evaluation": page_model,
    "Indicator Summary": page_indicators, "Student Details": page_student, "Export Reports": page_export,
}


def _streamlit_compat():
    """Streamlit 1.50+ replaces use_container_width with width="stretch"; support both."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return
    if (major, minor) < (1, 50):
        return
    for name in ("button", "download_button", "form_submit_button", "plotly_chart", "dataframe"):
        fn = getattr(st, name)
        if getattr(fn, "_edulens_wrapped", False):
            continue

        def wrapped(*a, _fn=fn, **k):
            if k.pop("use_container_width", False):
                k["width"] = "stretch"
            return _fn(*a, **k)
        wrapped._edulens_wrapped = True
        setattr(st, name, wrapped)


def main():
    st.set_page_config(page_title=f"{PRODUCT_NAME} | Student Analytics", page_icon=None, layout="wide",
                       initial_sidebar_state="expanded")
    _streamlit_compat()
    init_state()
    inject_css(st.session_state["authenticated"])
    if not st.session_state["authenticated"]:
        login_page()
        return
    sidebar()
    ROUTES.get(st.session_state["page"], page_upload)()


if __name__ == "__main__":
    main()
