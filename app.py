"""
EduLens - Student Performance & Dropout Analytics Dashboard (Streamlit frontend).

The backend (pipeline.py) is used unchanged. Everything shown here is computed
from the uploaded file and the pipeline's own outputs.

Sign-in uses accounts defined in .streamlit/secrets.toml (see SETUP.md);
passwords are stored only as salted PBKDF2-SHA256 hashes.
"""

import hashlib
import hmac
import html
import io
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.io.formats.style import Styler

from pipeline import (
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
# Constants (Academic Minimal: navy, warm ivory, sage green)
# ----------------------------------------------------------------------------
PRODUCT_NAME = "EduLens"
PRODUCT_TAG = "Student Performance & Dropout Analytics"

NAVY, NAVY2, NAVY_D, MUTED = "#23364D", "#33415C", "#18283B", "#6B7280"
SAGE, SAGE_D, AMBER, TERRA = "#829783", "#829783", "#C8A24E", "#B65A4B"
LINE = "#E5E5DF"

TIERS = ["High-Performing", "Moderate", "At-Risk"]
TIER_COLORS = {"High-Performing": SAGE, "Moderate": AMBER, "At-Risk": TERRA}
GRADES = ["O", "A", "B", "F"]
GRADE_COLORS = {"O": NAVY, "A": SAGE, "B": AMBER, "F": TERRA, "Unknown": "#A8A29A"}
HEAT = [[0, "#F7F6F0"], [0.5, "#B9C7B9"], [1, "#23364D"]]
GRADE_RULE = "O: score 75 or above · A: 60 to under 75 · B: 50 to under 60 · F: under 50"

MIN_RECORDS = 4          # K-Means with K = 3 needs more than 3 students
RECOMMENDED_RECORDS = 30
RECORD_RULE = (
    f"Minimum {MIN_RECORDS} student records (K-Means with K = {N_CLUSTERS} needs more than "
    f"{N_CLUSTERS}); {RECOMMENDED_RECORDS} or more recommended for stable groups."
)

# Analysis pages shown as the horizontal navigation bar (upload lives in the sidebar).
PAGES = [
    ("Data Validation", "File checks, preview and processing details."),
    ("Class Overview", "Quick summary of the class."),
    ("Performance", "Grades, scores and engagement indicators."),
    ("At-Risk Students", "Identify students needing attention."),
    ("Cluster Analysis", "Understand how students were grouped."),
    ("Model Evaluation", "Review clustering quality."),
    ("Indicator Summary", "Compare indicators across groups."),
    ("Student Details", "View one student's profile."),
    ("Export Reports", "Download results as CSV."),
]
PAGE_NAMES = [p for p, _ in PAGES]
DEFAULT_PAGE = PAGE_NAMES[0]

# Role-based permissions (enforced in the router and on the Export page).
# Both roles see every analysis page; they differ in which files they may download.
ROLES = {
    "Faculty Member": {
        "pages": PAGE_NAMES,
        "exports": {"results", "intervention", "summary"},
        "blurb": "All pages; may download student results, intervention list and class summary.",
    },
    "Course Coordinator": {
        "pages": PAGE_NAMES,
        "exports": {"intervention", "summary"},
        "blurb": "All pages; may download the intervention list and class summary only.",
    },
}

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
        "name": "Academic records (MTech)",
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
        "name": "Classroom engagement (xAPI)",
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
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');
:root{--navy:#23364D;--navy-d:#18283B;--ivory:#F7F6F0;--card:#FFFFFF;--line:#E5E5DF;--sage:#829783;--sage-l:#E8EEE7;--text:#293444;--muted:#6B7280;--amber:#C8A24E;--terra:#B65A4B;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--ivory);}
.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp li,.stApp td,.stApp th{font-family:Inter,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
.stApp{color:var(--text);line-height:1.55;}
.ph-title,.brand-name,.login-title,.login-form-title,.kpi-value{font-family:"Source Serif 4",Georgia,"Times New Roman",serif;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important;}
header[data-testid="stHeader"]{background:transparent;height:2.5rem;}
.main .block-container,[data-testid="stMainBlockContainer"]{max-width:1400px;padding:3.2rem 1.5rem 2rem;}
[data-testid="stVerticalBlock"]{gap:.75rem;}
.stApp label p,[data-testid="stWidgetLabel"] p{font-size:.84rem;font-weight:600;color:var(--text);line-height:1.4;}
[data-testid="stWidgetLabel"]{margin-bottom:.15rem;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{font-size:.82rem;line-height:1.55;color:#566070 !important;opacity:1 !important;}
[data-baseweb="tag"]{max-width:100% !important;}
[data-baseweb="tag"] span{font-size:.8rem;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.6;}
/* Streamlit gives markdown blocks a -1rem bottom margin that makes custom HTML blocks overlap the next element; neutralise it */
[data-testid="stMarkdownContainer"]{margin-bottom:0 !important;}
[data-testid="stMarkdownContainer"] > *:last-child{margin-bottom:0 !important;}

/* page header */
.ph{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:.3rem 1.5rem;margin:.1rem 0 0;}
.ph-title{font-size:1.6rem;font-weight:700;color:var(--navy);line-height:1.25;}
.ph-sub{font-size:.9rem;color:var(--muted);margin-top:.15rem;line-height:1.5;}
.ph-meta{font-size:.78rem;color:var(--muted);text-align:right;line-height:1.4;}

/* top navigation bar: one native button per page; wraps onto a second row on narrow screens so no page is ever hidden */
.st-key-navbar{border-bottom:1px solid var(--line);padding-bottom:.55rem;}
.st-key-navbar [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;gap:.4rem .35rem !important;}
.st-key-navbar [data-testid="stColumn"],.st-key-navbar [data-testid="column"]{flex:0 0 auto !important;width:auto !important;min-width:0 !important;}
.st-key-navbar button{white-space:nowrap;min-height:2.2rem;padding:.3rem .7rem;font-weight:600;border-radius:8px;}
.st-key-navbar button p{font-size:.82rem;white-space:nowrap;}
.st-key-navbar button[kind="secondary"],.st-key-navbar [data-testid="stBaseButton-secondary"]{background:var(--card);border:1px solid var(--line);color:var(--text);}
.st-key-navbar button[kind="secondary"]:hover,.st-key-navbar [data-testid="stBaseButton-secondary"]:hover{background:var(--sage-l);border-color:var(--sage);color:var(--navy);}
.st-key-navbar button[kind="primary"],.st-key-navbar [data-testid="stBaseButton-primary"]{background:var(--navy);border:1px solid var(--navy);color:#fff;}
.st-key-navbar button[kind="primary"] p,.st-key-navbar [data-testid="stBaseButton-primary"] p{color:#fff;}

/* KPI cards */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:.7rem;}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.7rem .9rem;position:relative;overflow:hidden;}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent,var(--sage));}
.kpi-label{font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);line-height:1.3;}
.kpi-value{font-size:1.5rem;font-weight:700;color:var(--navy);line-height:1.2;margin-top:.2rem;overflow-wrap:anywhere;}
.kpi-sub{font-size:.76rem;color:var(--muted);margin-top:.15rem;line-height:1.4;}

/* callouts and cards */
.callout{border-radius:8px;padding:.7rem .95rem;font-size:.88rem;line-height:1.6;border:1px solid;}
.callout ul{margin:.3rem 0 0 1.1rem;padding:0;}
.callout.info,.callout.ok{background:var(--sage-l);border-color:#CBD7CA;color:#2F4A37;}
.callout.warn{background:#F8F1DC;border-color:#E8D9A8;color:#6B4E12;}
.callout.err{background:#F7E6E3;border-color:#E3BBB4;color:#7A2F26;}
.callout.note{background:#EFEFE8;border-color:var(--line);color:#3F4A5E;}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem;}
.card h4{margin:0 0 .3rem;color:var(--navy);font-size:.95rem;font-weight:700;line-height:1.35;}
.card p,.card li{color:#4A5568;font-size:.86rem;line-height:1.6;margin:0;}
.card ul{margin:.3rem 0 0 1.1rem;padding:0;}
.section-title{color:var(--navy);font-size:1rem;font-weight:700;margin:.3rem 0 0;line-height:1.4;}
.empty{color:var(--muted);text-align:center;padding:1.6rem .8rem;font-size:.88rem;background:var(--card);border:1px solid var(--line);border-radius:8px;}
.badge{display:inline-block;border-radius:999px;padding:.15rem .7rem;font-size:.74rem;font-weight:700;}
.b-ok{background:var(--sage-l);color:#2F4A37;}.b-bad{background:#F7E6E3;color:#7A2F26;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--line);}
[data-testid="stSidebarUserContent"]{padding:1rem 1.1rem 1.2rem !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.8rem;}
.brand{display:flex;align-items:center;gap:.75rem;padding:.2rem 0 1rem;border-bottom:1px solid var(--line);}
.brand-logo{flex:0 0 40px;width:40px;height:40px;border-radius:9px;background:var(--navy);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.95rem;font-family:"Source Serif 4",Georgia,serif;}
.brand-text{min-width:0;}
.brand-name{font-weight:700;font-size:1.2rem;color:var(--navy);line-height:1.2;}
.brand-tag{font-size:.74rem;color:var(--muted);line-height:1.35;margin-top:.1rem;}
.side-card{background:var(--ivory);border:1px solid var(--line);border-radius:8px;padding:.65rem .85rem;}
.side-label{font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);line-height:1.3;}
.side-value{font-size:.95rem;font-weight:600;color:var(--navy);margin-top:.1rem;line-height:1.35;overflow-wrap:anywhere;}
.side-sub{font-size:.8rem;color:var(--muted);line-height:1.4;}
.status{font-size:.84rem;line-height:1.5;border-radius:8px;padding:.55rem .75rem;border:1px solid var(--line);background:var(--ivory);color:var(--text);overflow-wrap:anywhere;}
.status.ok{background:var(--sage-l);border-color:#CBD7CA;color:#2F4A37;}
.status.warn{background:#F8F1DC;border-color:#E8D9A8;color:#6B4E12;}
.status.err{background:#F7E6E3;border-color:#E3BBB4;color:#7A2F26;}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.45rem;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{padding:.8rem;border-radius:8px;}
[data-testid="stSidebar"] .stButton > button{min-height:2.4rem;}

/* login */
.login-hero{background:var(--navy-d);border-radius:12px;padding:2.2rem 2rem;min-height:440px;}
.login-hero .brand{border-bottom:0;padding-bottom:0;}
.login-hero .brand-name{color:#F7F6F0 !important;font-size:1.3rem;}
.login-hero .brand-logo{background:var(--sage);}
.login-hero .brand-tag{color:#B8C0CF !important;}
.login-title{color:#F7F6F0 !important;font-size:1.75rem;font-weight:700;line-height:1.3;margin:2.2rem 0 .8rem;}
.login-desc{color:#D5DAE3 !important;font-size:.95rem;line-height:1.65;}
.login-point{color:#E8EBF1 !important;font-size:.9rem;margin-top:.9rem;padding-left:.9rem;border-left:3px solid var(--sage);line-height:1.5;}
.login-form-title{color:var(--navy);font-size:1.5rem;font-weight:700;margin-top:1.4rem;line-height:1.3;}
.login-form-sub{color:var(--muted);font-size:.9rem;margin:.25rem 0 .7rem;line-height:1.5;}

/* controls */
.stButton > button,.stDownloadButton > button,[data-testid="stFormSubmitButton"] > button{border-radius:8px;font-weight:600;border:1px solid var(--line);}
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"],[data-testid="stFormSubmitButton"] > button[kind="primary"]{background:var(--navy);border-color:var(--navy);color:#fff;}
.stButton > button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover{background:var(--navy-d);border-color:var(--navy-d);color:#fff;}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:.45rem .85rem;color:var(--muted);}
.stTabs [data-baseweb="tab"] p{font-size:.88rem;}
.stTabs [aria-selected="true"]{color:var(--navy) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--navy);}
[data-testid="stFileUploaderDropzone"]{background:var(--card);border:1.5px dashed var(--sage);border-radius:8px;}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:2px 4px;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden;}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line);border-radius:8px;}
[data-testid="stExpander"] summary{padding:.5rem .9rem;}
[data-testid="stExpander"] summary p{font-size:.9rem;font-weight:600;}
</style>
"""

HIDE_SIDEBAR = """<style>[data-testid="stSidebar"],[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{display:none !important;}</style>"""


def inject_css(logged_in):
    st.markdown(CSS + ("" if logged_in else HIDE_SIDEBAR), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def page_header(title, sub, A=None):
    meta = ""
    if A is not None:
        meta = (f'<div class="ph-meta">{esc(A["name"])} · {A["n"]} students · '
                f'{esc(SCHEMAS[A["kind"]]["name"])}</div>')
    st.markdown(f'<div class="ph"><div><div class="ph-title">{esc(title)}</div>'
                f'<div class="ph-sub">{esc(sub)}</div></div>{meta}</div>', unsafe_allow_html=True)


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


def label(col):
    return DISPLAY.get(col, col)


def pct(a, b):
    return f"{(a / b * 100):.1f}%" if b else "0.0%"


def request_nav(page):
    """Switch page. Works from callbacks and from normal script flow (no widget owns this key)."""
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
    st.dataframe(df if isinstance(df, Styler) else safe_df(df), **kw)


def fit_h(rows, cap=300):
    """Compact table height: fits the rows, capped so pages stay short."""
    return int(min(cap, 40 + 35 * rows))


def style_fig(fig, h=290, title=None, legend=False):
    fig.update_layout(
        template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#4A5568"),
        margin=dict(t=46 if title else 16, b=100 if legend else 38, l=48, r=16),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=LINE),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.34, x=0, title_text="", font=dict(size=11)))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=NAVY)))
    fig.update_xaxes(gridcolor="#EDEDE6", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EDEDE6", zeroline=False, linecolor=LINE)
    return fig


def plot(fig, key=None):
    """Render a figure; None means there is nothing to draw."""
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
            "from one schema (see 'Supported datasets' in the sidebar).")
        return v
    if v["n"] < MIN_RECORDS:
        v["errors"].append(f"Only {v['n']} records found. {RECORD_RULE}")
    elif v["n"] < RECOMMENDED_RECORDS:
        v["warnings"].append(f"Small cohort ({v['n']} records): groups can change a lot with a few students. "
                             f"{RECOMMENDED_RECORDS}+ recommended.")
    sch = SCHEMAS[kind]
    for col in sch["features"]:
        found = col in cleaned.columns
        role = "Categorical (text)" if (kind == "xapi" and col in XAPI_BINARY_MAP) else "Numeric"
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


def build_analysis(name, raw, progress=None):
    """Run the unchanged backend and assemble everything the pages need."""
    step = progress or (lambda p, t: None)
    step(10, "Reading column names...")
    cleaned = clean_columns(raw).reset_index(drop=True)
    step(30, "Cleaning values, filling gaps and scaling indicators...")
    prep = preprocess(cleaned)
    step(55, f"Grouping students with K-Means (K = {N_CLUSTERS})...")
    df, _model, metrics = run_kmeans(prep, N_CLUSTERS)
    step(80, "Deriving grades, risk flags and indicators...")
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
    step(100, "Done.")

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
# Authentication (credentials live in .streamlit/secrets.toml, never in this file)
# ----------------------------------------------------------------------------
HASH_SCHEME = "pbkdf2_sha256"
MAX_FAILED, LOCK_SECONDS = 5, 30
_DUMMY_HASH = f"{HASH_SCHEME}$600000${'00' * 16}${'00' * 32}"


def verify_password(password, stored):
    """Check a password against 'pbkdf2_sha256$iterations$salt_hex$hash_hex' in constant time."""
    try:
        scheme, iters, salt_hex, hash_hex = str(stored).split("$")
        if scheme != HASH_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


def load_accounts():
    """Read accounts from st.secrets['edulens']['users']. Returns {} if none are configured."""
    accounts = {}
    try:
        users = st.secrets["edulens"]["users"]
        for uname, rec in dict(users).items():
            rec = dict(rec)
            roles = [r for r in rec.get("roles", []) if r in ROLES]
            if rec.get("password_hash") and roles:
                accounts[str(uname).strip().lower()] = {
                    "name": str(rec.get("name") or uname), "roles": roles, "hash": str(rec["password_hash"])}
    except Exception:
        return {}
    return accounts


def authenticate(username, password, role):
    """Returns (account or None, error message). The same message is used for every credential failure."""
    accounts = load_accounts()
    if not accounts:
        return None, "no_accounts"
    acct = accounts.get(username.strip().lower())
    ok = verify_password(password, acct["hash"] if acct else _DUMMY_HASH)   # dummy check keeps timing similar
    if acct and ok and role in acct["roles"]:
        return acct, None
    return None, "Incorrect username, password or role."


def login_page():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            f'<div class="login-hero"><div class="brand"><div class="brand-logo">EL</div><div>'
            f'<div class="brand-name">{PRODUCT_NAME}</div><div class="brand-tag">{PRODUCT_TAG}</div></div></div>'
            '<div class="login-title">See which students need attention, and why.</div>'
            '<div class="login-desc">Upload a class dataset and get an overview of performance, relative risk groups '
            'and an intervention list. Built to support faculty decisions, not to replace them.</div>'
            '<div class="login-point">Class overview with grade and risk distributions</div>'
            '<div class="login-point">Intervention list with the indicators behind each flag</div>'
            '<div class="login-point">Cluster analysis explained in plain language, with CSV reports</div></div>',
            unsafe_allow_html=True)
    with right:
        st.markdown('<div class="login-form-title">Sign in</div>'
                    '<div class="login-form-sub">Use the account issued to you by your administrator.</div>',
                    unsafe_allow_html=True)
        no_accounts = not load_accounts()
        with st.form("login_form"):
            role = st.selectbox("Role", list(ROLES))
            user = st.text_input("Username", placeholder="Your username")
            pw = st.text_input("Password", type="password", placeholder="Your password")
            go_btn = st.form_submit_button("Sign in", type="primary", use_container_width=True,
                                           disabled=no_accounts)
        st.caption("Faculty Member: all pages and all downloads. Course Coordinator: all pages; downloads limited to "
                   "the intervention list and class summary.")
        if no_accounts:
            st.error("No accounts are configured yet. Create one with make_user.py and add it to "
                     ".streamlit/secrets.toml (see SETUP.md).")
        if not go_btn:
            return
        wait = st.session_state.get("lock_until", 0) - time.time()
        if wait > 0:
            st.error(f"Too many failed attempts. Try again in {int(wait) + 1} seconds.")
            return
        problems = []
        if not user.strip():
            problems.append("Enter your username.")
        if not pw:
            problems.append("Enter your password.")
        if problems:
            for p in problems:
                st.error(p)
            return
        acct, err = authenticate(user, pw, role)
        if acct is None:
            fails = st.session_state.get("login_fails", 0) + 1
            st.session_state["login_fails"] = fails
            if fails >= MAX_FAILED:
                st.session_state.update(login_fails=0, lock_until=time.time() + LOCK_SECONDS)
                st.error(f"Too many failed attempts. Locked for {LOCK_SECONDS} seconds.")
            else:
                st.error(err)
            return
        for k in ("login_fails", "lock_until"):
            st.session_state.pop(k, None)
        st.session_state.update(authenticated=True, username=user.strip().lower(), user=acct["name"], role=role,
                                page=DEFAULT_PAGE)
        st.rerun()


# ----------------------------------------------------------------------------
# Session state, shell
# ----------------------------------------------------------------------------
def init_state():
    for k, v in {"authenticated": False, "upload": None, "analysis": None, "page": DEFAULT_PAGE,
                 "file_sig": None, "run_status": None}.items():
        st.session_state.setdefault(k, v)


def logout():
    """Clear everything: authentication, role, uploaded data and results."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def get_analysis():
    return st.session_state.get("analysis")


def allowed_pages():
    return ROLES.get(st.session_state.get("role"), {}).get("pages", [])


def can_export(kind):
    return kind in ROLES.get(st.session_state.get("role"), {}).get("exports", set())


def current_status():
    """(css class, message) describing the dataset: none / uploaded / running result / done / invalid / failed."""
    A, up, rs = get_analysis(), st.session_state.get("upload"), st.session_state.get("run_status")
    if A is not None:
        return "ok", f"Analysis complete: {A['n']} students from {A['name']}."
    if rs:
        return rs
    if up is not None and up.get("error"):
        return "err", f"Invalid dataset: {up['error']}"
    if up is not None and up.get("val") and up["val"]["errors"]:
        return "err", "Invalid dataset: " + up["val"]["errors"][0]
    if st.session_state.get("file_sig"):
        return "warn", "Dataset uploaded but not analysed. Press Run Analysis."
    return "", "No dataset uploaded."


def no_data_message():
    kind, msg = current_status()
    callout({"ok": "ok", "warn": "warn", "err": "err"}.get(kind, "note"),
            f"<b>{esc(msg)}</b> Use the sidebar to upload a CSV and press Run Analysis; every number on the "
            "analysis pages is computed from your data.")


def require_data(title, sub):
    A = get_analysis()
    page_header(title, sub, A)
    if A is None:
        no_data_message()
        return None
    return A


def run_analysis(f):
    """Validate the uploaded file, run the unchanged backend, keep results in session state."""
    st.session_state["analysis"] = None            # a new run invalidates earlier results
    st.session_state["run_status"] = None
    if f is None:
        st.session_state["run_status"] = ("err", "Please upload a CSV file first.")
        return
    raw, err = read_csv_bytes(f.getvalue())
    if err:
        st.session_state["upload"] = {"name": f.name, "raw": None, "error": err, "val": None}
        st.session_state["run_status"] = ("err", f"Invalid dataset: could not read {f.name}. {err}")
        return
    raw = raw.reset_index(drop=True)
    val = validate_upload(raw)
    st.session_state["upload"] = {"name": f.name, "raw": raw, "error": None, "val": val}
    if val["errors"]:
        st.session_state["run_status"] = ("err", "Invalid dataset: " + " ".join(val["errors"]) +
                                          " Details are on Data Validation.")
        return
    bar = st.progress(0, text="Starting...")
    try:
        A = build_analysis(f.name, raw, progress=lambda p, t: bar.progress(p, text=t))
    except Exception as exc:
        bar.empty()
        st.session_state["run_status"] = ("err", f"Analysis failed: {exc}")
        return
    st.session_state["analysis"] = A
    st.session_state["page"] = DEFAULT_PAGE
    st.rerun()


def sidebar():
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-logo">EL</div><div class="brand-text">'
            f'<div class="brand-name">{PRODUCT_NAME}</div><div class="brand-tag">{PRODUCT_TAG}</div></div></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-card"><div class="side-label">Welcome</div>'
            f'<div class="side-value">{esc(st.session_state.get("user", ""))}</div>'
            f'<div class="side-sub">{esc(st.session_state.get("role", ""))}</div></div>', unsafe_allow_html=True)

        f = st.file_uploader("Dataset (CSV file)", type=["csv"], key="uploader")
        sig = None if f is None else (getattr(f, "file_id", None) or (f.name, getattr(f, "size", 0)))
        if f is not None and sig != st.session_state.get("file_sig"):
            # A different file was uploaded: old results no longer match it.
            st.session_state.update(file_sig=sig, analysis=None, upload=None, run_status=None, page=DEFAULT_PAGE)
        elif f is None and st.session_state.get("file_sig") is not None:
            st.session_state["file_sig"] = None
        if st.button("Run Analysis", type="primary", use_container_width=True, key="run_analysis"):
            run_analysis(f)
        kind, msg = current_status()
        st.markdown(f'<div class="status {kind}">{esc(msg)}</div>', unsafe_allow_html=True)

        with st.expander("Supported datasets"):
            st.markdown(
                "**A. Academic Records (MTech)** needs at least 3 of: " + ", ".join(MTECH_FEATURE_COLS) +
                " (numbers only). Optional: Student ID or Roll Number, QUIZ 1 to QUIZ 6.\n\n"
                "**B. Classroom Engagement (xAPI)** needs at least 3 of: raisedhands, VisITedResources, "
                "AnnouncementsView, Discussion (numbers); StudentAbsenceDays (Under-7 or Above-7); "
                "ParentAnsweringSurvey (No or Yes); ParentschoolSatisfaction (Bad or Good). Optional: Class (H, M, L), "
                "Topic, Semester, GradeID, SectionID.\n\n"
                f"Column names must match exactly (extra spaces are trimmed). {RECORD_RULE}")
        if st.button("Log out", key="logout_btn", use_container_width=True):
            logout()
            st.rerun()


def navbar(allowed):
    """Horizontal top navigation: one native button per analysis page; the active page is filled."""
    cur = st.session_state.get("page")
    with st.container(key="navbar"):
        cols = st.columns(len(allowed))
        for i, (col, p) in enumerate(zip(cols, allowed)):
            col.button(p, key=f"nav_{i}", type="primary" if p == cur else "secondary",
                       on_click=request_nav, args=(p,))


# ----------------------------------------------------------------------------
# Page: Data Validation
# ----------------------------------------------------------------------------
def page_validation():
    A, up = get_analysis(), st.session_state.get("upload")
    page_header("Data Validation", "File checks, a preview of the uploaded records, and how values were processed.", A)
    if up is None:
        no_data_message()
        return
    if up["error"]:
        callout("err", f"<b>Could not read {esc(up['name'])}.</b> {esc(up['error'])}")
        return
    v, raw = up["val"], up["raw"]
    ok = not v["errors"]
    badge = ('<span class="badge b-ok">Passed pipeline requirements</span>' if ok
             else '<span class="badge b-bad">Cannot be analysed</span>')
    schema = SCHEMAS[v["kind"]]["name"] if v["kind"] else "unrecognised columns"
    st.markdown(f'{badge} &nbsp;<span style="font-size:.86rem;color:{MUTED}">{esc(up["name"])} · recognised as '
                f'{esc(schema)}</span>', unsafe_allow_html=True)
    cells = v["n"] * max(len(v["used"]), 1)
    kpis([
        ("Records", f"{v['n']:,}", "student rows", SAGE_D),
        ("Columns in file", v["n_cols"], "total", MUTED),
        ("Indicators used", len(v["used"]), "recognised by pipeline", NAVY2),
        ("Missing values", f"{v['missing_cells']:,}", f"{pct(v['missing_cells'], cells)} of indicator cells", AMBER),
        ("Duplicate rows", v["dup_rows"], f"{v['dup_ids']} repeated IDs", AMBER if v["dup_rows"] else SAGE_D),
    ])
    if v["errors"]:
        callout("err", "<b>This dataset cannot be analysed.</b><ul>" +
                "".join(f"<li>{esc(e)}</li>" for e in v["errors"]) + "</ul>")
    if v["warnings"]:
        with st.expander(f"{len(v['warnings'])} item(s) to review", expanded=len(v["warnings"]) <= 2):
            st.markdown("\n".join(f"- {w}" for w in v["warnings"]))
    a, b = st.columns([1, 1.25])
    with a:
        section_title("Column validation")
        if v["rows"]:
            show_df(pd.DataFrame(v["rows"]), height=fit_h(len(v["rows"]), 330))
            st.caption("Missing = empty cells. Invalid = values not readable as the expected type. Both are filled "
                       "with the column median during processing.")
        else:
            st.caption("No column check is available because the columns were not recognised.")
    with b:
        section_title("Dataset preview (as uploaded)")
        show_df(raw.head(10), height=fit_h(10, 330))
        st.caption(f"First {min(10, len(raw))} of {len(raw):,} rows, exactly as uploaded.")
    if A:
        with st.expander("Processing information"):
            st.markdown(
                f"Outliers are clipped to the 1st-99th percentile of each indicator, missing values are filled with "
                f"the column median, then values are standardised for K-Means (K = {A['metrics']['n_clusters']}). "
                f"Analysed at {esc(A['built'])}.")
            show_df(trail_table(A), height=fit_h(len(A["trail"]), 330))
        st.button("Continue to Class Overview", type="primary", on_click=request_nav, args=("Class Overview",),
                  key="val_next")
    elif ok:
        st.caption("Press Run Analysis in the sidebar to process this file.")


def trail_table(A):
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
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Chart builders
# ----------------------------------------------------------------------------
def fig_grade_bar(df, title="Grade distribution", h=270):
    c = df["Grade"].value_counts()
    order = [g for g in GRADES + ["Unknown"] if g in c.index or g in GRADES]
    d = pd.DataFrame({"Grade": order, "Students": [int(c.get(g, 0)) for g in order]})
    d["Share"] = (d["Students"] / max(len(df), 1) * 100).round(1)
    fig = px.bar(d, x="Grade", y="Students", color="Grade", color_discrete_map=GRADE_COLORS, text="Students",
                 custom_data=["Share"])
    fig.update_traces(hovertemplate="Grade %{x}<br>%{y} students (%{customdata[0]}%)<extra></extra>",
                      textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, max(int(d["Students"].max()), 1) * 1.2])
    return style_fig(fig, h, title)


def fig_tier_donut(df, title="Risk tier distribution", h=270):
    c = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int)
    if c.sum() == 0:
        return None
    fig = go.Figure(go.Pie(labels=list(c.index), values=list(c.values), hole=0.55, sort=False,
                           marker=dict(colors=[TIER_COLORS[t] for t in c.index]),
                           textinfo="label+percent", hovertemplate="%{label}<br>%{value} students<extra></extra>"))
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    return style_fig(fig, h, title)


def fig_score_hist(A, color="cluster", title=None, h=270):
    df, sc = A["df"], A["score_col"]
    if not has_values(df, sc):
        return None
    fig = px.histogram(df, x=sc, color=color, nbins=20, barmode="overlay", opacity=0.75,
                       color_discrete_map={**TIER_COLORS, **GRADE_COLORS},
                       category_orders={"cluster": TIERS, "Grade": GRADES},
                       labels={sc: A["score_label"], "cluster": "Tier"})
    fig.update_layout(yaxis_title="Students", xaxis_title=A["score_label"])
    fig.add_vline(x=float(df[sc].mean()), line_dash="dash", line_color=NAVY,
                  annotation_text=f"Mean {df[sc].mean():.1f}", annotation_position="top")
    return style_fig(fig, h, title or f"{A['score_label']} distribution", legend=True)


def fig_box(df, y, ylabel, by="cluster", title=None, order=TIERS, cmap=None, h=270):
    if not has_values(df, y):
        return None
    fig = px.box(df, x=by, y=y, color=by, points="all", color_discrete_map=cmap or TIER_COLORS,
                 category_orders={by: order}, hover_data=[c for c in ["Student ID"] if c in df.columns])
    fig.update_layout(xaxis_title="Risk tier" if by == "cluster" else by, yaxis_title=ylabel)
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    return style_fig(fig, h, title or f"{ylabel} by {'risk tier' if by == 'cluster' else by}")


def grade_caveat(A):
    """One visible line plus an optional expander (keeps pages short)."""
    if A["kind"] == "xapi":
        st.caption("Grade here is a band of the average engagement count, not an exam grade, and tiers are relative "
                   "to this class; both support faculty decisions and do not predict outcomes.")
        long = ("The file holds engagement counts, not exam marks. Grade is a band of the average engagement count "
                f"({GRADE_RULE}), so Grade F means low recorded engagement relative to those fixed cut-offs and can "
                "cover a large share of the class.")
    else:
        st.caption("Grades come from the AVG column and tiers are relative to this class; both support faculty "
                   "decisions and do not predict outcomes.")
        long = (f"Grade bands are derived from the AVG column: {GRADE_RULE}. Grade and risk tier are separate "
                "labels: a student can be Grade F and still sit in the Moderate tier.")
    with st.expander("How grades and tiers are derived"):
        st.markdown(long + " Tiers compare students within this upload; see Cluster Analysis for the exact method.")


# ----------------------------------------------------------------------------
# Page: Class overview
# ----------------------------------------------------------------------------
def page_overview():
    A = require_data("Class Overview", "A snapshot of the class: risk tiers, grades and score spread.")
    if A is None:
        return
    df, n = A["df"], A["n"]
    cnt = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int)
    f_cnt = int((df["Grade"] == "F").sum())
    flagged = int(df["In intervention list"].sum())
    kpis([
        ("Total students", f"{n:,}", "in uploaded file", SAGE_D),
        ("At-Risk tier", int(cnt["At-Risk"]), pct(cnt["At-Risk"], n) + " of class", TIER_COLORS["At-Risk"]),
        ("Moderate tier", int(cnt["Moderate"]), pct(cnt["Moderate"], n) + " of class", TIER_COLORS["Moderate"]),
        ("High-Performing", int(cnt["High-Performing"]), pct(cnt["High-Performing"], n) + " of class",
         TIER_COLORS["High-Performing"]),
        (f"Mean {A['score_label']}", f"{df[A['score_col']].mean():.1f}", "processed values", NAVY2),
        ("Grade F", f_cnt, pct(f_cnt, n) + " of class", TERRA),
        ("Intervention list", flagged, "At-Risk tier or Grade F", NAVY),
    ])
    grade_caveat(A)
    a, b, c = st.columns(3)
    with a:
        plot(fig_grade_bar(df), "ov_grade")
    with b:
        plot(fig_tier_donut(df), "ov_tier")
    with c:
        plot(fig_score_hist(A), "ov_hist")
    a, b = st.columns([3, 2])
    with a:
        section_title("Class statistics (processed values)")
        stats = df[A["features"] + [A["score_col"]]].describe().T[["mean", "50%", "std", "min", "max"]].round(2)
        stats.index = [label(c) if c != A["score_col"] else A["score_label"] + " (derived)" for c in stats.index]
        stats.columns = ["Mean", "Median", "Std. dev.", "Min", "Max"]
        show_df(stats.reset_index().rename(columns={"index": "Indicator"}), height=fit_h(len(stats), 280))
        st.caption("Processed = clipped to the 1st-99th percentile and median-filled (details on Data Validation).")
    with b:
        plot(fig_box(df, A["score_col"], A["score_label"], h=330), "ov_box")
    st.button("Review students needing attention", type="primary", on_click=request_nav, args=("At-Risk Students",),
              key="ov_to_risk")


# ----------------------------------------------------------------------------
# Page: Performance
# ----------------------------------------------------------------------------
def page_performance():
    A = require_data("Performance", "How grades, scores and engagement indicators are distributed across the class.")
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
        a, b, c = st.columns(3)
        with a:
            m = df.groupby(["cluster", "Grade"]).size().reset_index(name="Students")
            fig = px.bar(m, x="cluster", y="Students", color="Grade", barmode="stack",
                         color_discrete_map=GRADE_COLORS, category_orders={"cluster": TIERS, "Grade": GRADES})
            fig.update_layout(xaxis_title="Risk tier")
            plot(style_fig(fig, 300, "Grades within each risk tier", legend=True), "pf_gt")
        with b:
            plot(fig_box(df, A["score_col"], A["score_label"], by="Grade", order=GRADES, cmap=GRADE_COLORS,
                         title=f"{A['score_label']} by grade", h=300), "pf_sg")
        with c:
            plot(fig_score_hist(A, color="Grade", h=300), "pf_hist")

    with tab["Key indicators"]:
        keys = [c for c in sch["key"] if has_values(df, c)]
        ind = st.selectbox("Explore another indicator", A["features"], format_func=label, key="pf_ind")
        cols = st.columns(3)
        for i, c in enumerate(keys[:2]):
            with cols[i]:
                plot(fig_box(df, c, label(c), h=300), f"pf_key_{c}")
        with cols[2]:
            if has_values(df, ind):
                fig = px.histogram(df, x=ind, color="cluster", nbins=20, barmode="overlay", opacity=0.75,
                                   color_discrete_map=TIER_COLORS, category_orders={"cluster": TIERS},
                                   labels={ind: label(ind), "cluster": "Tier"})
                fig.update_layout(yaxis_title="Students")
                plot(style_fig(fig, 300, f"{label(ind)} distribution", legend=True), "pf_ind_h")
            else:
                plot(None)
        if A["kind"] == "mtech":
            st.caption("The pipeline has no assignment-marks column; quiz results are in the Quiz performance tab.")

    if A["quiz_cols"]:
        with tab["Quiz performance"]:
            q = df.groupby("cluster")[A["quiz_cols"]].mean().reindex(TIERS)
            long = q.reset_index().melt(id_vars="cluster", var_name="Quiz", value_name="Mean score").dropna()
            fig = px.line(long, x="Quiz", y="Mean score", color="cluster", markers=True,
                          color_discrete_map=TIER_COLORS, category_orders={"cluster": TIERS},
                          labels={"cluster": "Tier"}) if len(long) else None
            plot(style_fig(fig, 320, "Mean quiz score by risk tier", legend=True) if fig else None, "pf_quiz")
            st.caption("Quiz columns are shown exactly as uploaded; empty quiz cells are left out of the averages.")

    if groupers:
        with tab["Course comparison"]:
            g = st.selectbox("Compare by", groupers, key="pf_group")
            d = df.groupby(g).agg(Students=(A["id_col"], "count"), Mean=(A["score_col"], "mean"),
                                  AtRisk=("cluster", lambda s: (s == "At-Risk").mean() * 100)).reset_index()
            a, b = st.columns(2)
            with a:
                fig = px.bar(d, x=g, y="Mean", text=d["Mean"].round(1), color_discrete_sequence=[SAGE_D],
                             hover_data={"Students": True})
                fig.update_layout(yaxis_title=A["score_label"])
                plot(style_fig(fig, 300, f"Mean {A['score_label']} by {g}"), "pf_cg1")
            with b:
                fig = px.bar(d, x=g, y="AtRisk", text=d["AtRisk"].round(1), color_discrete_sequence=[TERRA],
                             hover_data={"Students": True})
                fig.update_layout(yaxis_title="% of students in At-Risk tier")
                plot(style_fig(fig, 300, f"At-Risk share by {g}"), "pf_cg2")
            st.caption("Group sizes can be small; check the Students count in the tooltip before drawing conclusions.")


# ----------------------------------------------------------------------------
# Page: At-risk students
# ----------------------------------------------------------------------------
def style_tier(styler, cols):
    colors = {"At-Risk": "color:#7A2F26;font-weight:700", "Moderate": "color:#7A5A12;font-weight:600",
              "High-Performing": "color:#2F4A37;font-weight:600"}
    fn = lambda v: colors.get(v, "")
    return (styler.map(fn, subset=cols) if hasattr(styler, "map") else styler.applymap(fn, subset=cols))


def page_risk():
    A = require_data("At-Risk Students", "Students who may benefit from a conversation or extra support.")
    if A is None:
        return
    df, sch, idc = A["df"], SCHEMAS[A["kind"]], A["id_col"]
    fl = df[df["In intervention list"]]
    both = int((fl["Flag basis"] == "At-Risk tier + Grade F").sum())
    f_only = int((fl["Flag basis"] == "Grade F only").sum())
    f_mod = int(((fl["Grade"] == "F") & (fl["cluster"] == "Moderate")).sum())
    kpis([
        ("Students flagged", len(fl), pct(len(fl), A["n"]) + " of class", NAVY),
        ("At-Risk tier", int((df["cluster"] == "At-Risk").sum()), "cluster label", TIER_COLORS["At-Risk"]),
        ("Grade F", int((df["Grade"] == "F").sum()), "derived grade", TERRA),
        ("Both flags", both, "highest priority", NAVY2),
        ("Grade F, not At-Risk tier", f_only, f"{f_mod} are in Moderate tier", AMBER),
    ])
    st.caption("Listed: every student in the At-Risk tier or with Grade F (a Grade F student is never left out because "
               "their cluster is Moderate). These indicators support faculty decisions and do not predict individual "
               "outcomes.")
    with st.expander("How to read this list"):
        st.markdown(
            "'Possible indicators' only point out values that are low compared with the rest of this class (below its "
            "lower quartile). A low value is an association in this data, not a proven cause. The derived score is "
            "not listed as a reason, because Grade is calculated from it."
            + (" This dataset has no attendance or quiz-attempt columns, so the closest available indicators "
               "(participation, resource use, absence days) are shown." if A["kind"] == "xapi" else ""))
        grade_caveat_text = ("Grade here is an engagement band, so Grade F can cover many students."
                             if A["kind"] == "xapi" else f"Grade bands: {GRADE_RULE}.")
        st.markdown(grade_caveat_text)
    if fl.empty:
        callout("ok", "No students are currently flagged.")
        return

    f1, f2, f3 = st.columns([2.2, 1.7, 1.2])
    basis = f1.multiselect("Flag basis", ["At-Risk tier + Grade F", "At-Risk tier only", "Grade F only"],
                           default=["At-Risk tier + Grade F", "At-Risk tier only", "Grade F only"], key="rk_basis")
    tiers = f2.multiselect("Risk tier", TIERS, default=TIERS, key="rk_tier")
    grades = f3.multiselect("Grade", GRADES, default=GRADES, key="rk_grade")

    cols_ind = [c for c in sch["table"] if c in df.columns]
    sort_opts = ["Priority (default)", "Derived score"] + [label(c) for c in cols_ind]
    s0, s1, s2, s3 = st.columns([1.2, 1.4, 1, 2])
    search = s0.text_input("Search student ID", key="rk_search", placeholder="Part of an ID")
    sort_by = s1.selectbox("Sort by", sort_opts, key="rk_sort")
    asc = s2.selectbox("Order", ["Ascending", "Descending"], key="rk_order") == "Ascending"
    s3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    only_reasons = s3.checkbox("Only students with an indicator below the class lower quartile", key="rk_reason")

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
    st.markdown(f"**{len(view)}** of {len(fl)} flagged students shown")
    if show.empty:
        callout("note", "No students match the current filters.")
    else:
        st.dataframe(style_tier(show.style.format(precision=1), ["Risk tier"]), use_container_width=True,
                     hide_index=True, height=fit_h(len(show), 370))
    st.caption("Processed values (clipped to the 1st-99th percentile, gaps median-filled). Original uploaded values "
               "are on Student Details.")
    with st.expander("Chart of the filtered students"):
        if show.empty:
            st.markdown('<div class="empty">No students to chart for the current filters.</div>',
                        unsafe_allow_html=True)
        else:
            gcount = view.groupby(["cluster", "Grade"]).size().reset_index(name="Students")
            fig = px.bar(gcount, x="cluster", y="Students", color="Grade", color_discrete_map=GRADE_COLORS,
                         category_orders={"cluster": TIERS, "Grade": GRADES}, barmode="stack")
            fig.update_layout(xaxis_title="Risk tier")
            st.plotly_chart(style_fig(fig, 260, "Filtered students by tier and grade", legend=True),
                            use_container_width=True, key="rk_chart")


# ----------------------------------------------------------------------------
# Page: Cluster analysis
# ----------------------------------------------------------------------------
def sil_note(s):
    if s < 0.25:
        return "Little clear structure: groups overlap a lot."
    if s < 0.5:
        return "Weak to moderate separation; treat tiers as a rough guide."
    return "Clearer separation between groups."


def db_note(d):
    return ("Lower is better. Above about 1: groups are wider than the gaps between them."
            if d > 1 else "Lower is better. Below 1: groups are fairly compact relative to their gaps.")


def metric_card(title, value, note):
    st.markdown(f'<div class="card" style="border-top:3px solid {AMBER};min-height:9.2rem"><div class="side-label">{esc(title)}</div>'
                f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub" style="margin-top:.2rem">{esc(note)}'
                '</div></div>', unsafe_allow_html=True)


def page_cluster():
    A = require_data("Cluster Analysis", "How the pipeline grouped students and how clear the grouping is.")
    if A is None:
        return
    df, m = A["df"], A["metrics"]
    extra = ""
    if A["kind"] == "xapi":
        extra = (" All indicators are treated as 'higher = stronger' in this ranking, including the absence-over-7-days "
                 "indicator, because the pipeline averages the normalised values without reversing any column.")
    st.caption("Tiers are named by ranking the groups on their average normalised score; they are relative to this "
               "class and are not fixed score bands, quartiles or dropout forecasts.")
    with st.expander("How groups and tier names are assigned"):
        st.markdown(
            "- **Step 1:** each indicator is cleaned (extreme values clipped, gaps filled with the median) and "
            f"standardised.\n- **Step 2:** K-Means splits students into {m['n_clusters']} groups of similar profiles.\n"
            "- **Step 3:** every indicator is rescaled to 0-1 across the class and each group gets the average of "
            "those values for its members. The group with the highest average is named High-Performing, the lowest "
            f"At-Risk, and the remaining one Moderate.{extra}")

    sil, dbi = m["silhouette"], m["davies_bouldin"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Silhouette score", f"{sil:.3f}", sil_note(sil) + " Range -1 to 1; higher = clearer.")
    with c2:
        metric_card("Davies-Bouldin index", f"{dbi:.3f}", db_note(dbi))
    with c3:
        metric_card("Clusters (K)", str(m["n_clusters"]), "Fixed by the pipeline (N_CLUSTERS).")
    with c4:
        metric_card("PCA variance (2D map)", f"{A['pca_variance'].sum() * 100:.1f}%",
                    "Share of variation kept by the 2D map; the rest is not visible.")
    st.caption("These measures describe how well separated the groups are, not prediction accuracy. Students near a "
               "boundary could reasonably belong to the neighbouring tier.")

    a, b, c = st.columns([1, 1, 1.5])
    with a:
        cd = df["cluster"].value_counts().reindex(TIERS).fillna(0).astype(int).reset_index()
        cd.columns = ["Tier", "Students"]
        fig = px.bar(cd, x="Tier", y="Students", color="Tier", color_discrete_map=TIER_COLORS, text="Students")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, max(int(cd["Students"].max()), 1) * 1.2])
        plot(style_fig(fig, 320, "Cluster distribution"), "cl_dist")
    with b:
        lab2tier = df.drop_duplicates("kmeans_label").set_index("kmeans_label")["cluster"].to_dict()
        sc = pd.DataFrame([(lab2tier.get(int(k), str(k)), v) for k, v in m["cluster_performance_scores"].items()],
                          columns=["Tier", "Average normalised score"]).set_index("Tier").reindex(TIERS).reset_index()
        fig = px.bar(sc, x="Tier", y="Average normalised score", color="Tier", color_discrete_map=TIER_COLORS,
                     text=sc["Average normalised score"].round(3))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_range=[0, 1.1])
        plot(style_fig(fig, 320, "Ranking score that names the tiers (0-1)"), "cl_rank")
    with c:
        hover = [x for x in [A["id_col"], A["score_col"], "Grade"] if x in df.columns]
        fig = px.scatter(df, x="pca_x", y="pca_y", color="cluster", color_discrete_map=TIER_COLORS,
                         category_orders={"cluster": TIERS}, hover_data=hover,
                         labels={"pca_x": "Principal component 1", "pca_y": "Principal component 2", "cluster": "Tier"})
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        plot(style_fig(fig, 320, "Students on a 2D map (PCA)", legend=True), "cl_pca")

    a, b = st.columns([1, 1.1])
    with a:
        mm = df[A["features"]].copy()
        rng = (mm.max() - mm.min()).replace(0, np.nan)
        norm = ((mm - mm.min()) / rng).fillna(0)
        heat = norm.groupby(df["cluster"]).mean().reindex(TIERS)
        heat.columns = [label(x) for x in heat.columns]
        fig = px.imshow(heat.values, x=list(heat.columns), y=list(heat.index), text_auto=".2f", aspect="auto",
                        color_continuous_scale=HEAT, zmin=0, zmax=1)
        fig.update_layout(coloraxis_colorbar=dict(title="0-1"))
        fig.update_xaxes(tickangle=-40)
        plot(style_fig(fig, 340, "Cluster comparison (average normalised value)"), "cl_heat")
    with b:
        section_title("Cluster characteristics (average processed values)")
        prof = df.groupby("cluster")[A["features"]].mean().reindex(TIERS).T
        prof.index = [label(x) for x in prof.index]
        prof["Class mean"] = df[A["features"]].mean().values
        show_df(prof.round(2).reset_index().rename(columns={"index": "Indicator"}), height=fit_h(len(prof), 300))
        sizes = df.groupby("cluster").agg(Students=(A["id_col"], "count"),
                                          GradeF=("Grade", lambda s: int((s == "F").sum()))).reindex(TIERS).reset_index()
        sizes.columns = ["Tier", "Students", "Grade F students"]
        show_df(sizes, height=fit_h(3))
    if "ari_vs_actual_class" in m:
        st.caption(f"Agreement with the file's own Class label (H/M/L): adjusted Rand index "
                   f"{m['ari_vs_actual_class']:.3f}, normalised mutual information {m['nmi_vs_actual_class']:.3f}. "
                   "Near 0 means little agreement beyond chance; this is agreement, not accuracy.")


# ----------------------------------------------------------------------------
# Page: Model evaluation
# ----------------------------------------------------------------------------
def page_model():
    A = require_data("Model Evaluation", "Cluster-quality checks. K-Means is unsupervised, so there is no accuracy "
                                         "score or training curve.")
    if A is None:
        return
    df, m = A["df"], A["metrics"]
    t1, t2, t3 = st.tabs(["Choice of K", "Score distribution" + (" & quiz trend" if A["quiz_cols"] else ""),
                          "Agreement with provided labels"])
    with t1:
        st.caption(f"The pipeline uses K = {m['n_clusters']}. The charts re-run the backend's evaluate_k_values to show "
                   "how other values of K compare on this data; this does not change the analysis.")
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
            specs = [("Inertia", "Inertia (lower = tighter; always falls as K grows)"),
                     ("Silhouette", "Silhouette (higher = clearer separation)"),
                     ("Davies-Bouldin", "Davies-Bouldin (lower = clearer separation)")]
            for col, (name, title) in zip(cols, specs):
                with col:
                    fig = px.line(k, x="K", y=name, markers=True, color_discrete_sequence=[SAGE_D])
                    fig.add_vline(x=m["n_clusters"], line_dash="dash", line_color=AMBER,
                                  annotation_text="K used", annotation_position="top")
                    fig.update_xaxes(dtick=1)
                    plot(style_fig(fig, 280, title), f"me_{name}")
            with st.expander("Values behind the charts"):
                show_df(k.round(4), height=fit_h(len(k)))
            st.caption("Different measures can disagree; use them as a sanity check, not as proof that a particular K "
                       "is correct.")
    with t2:
        a, b = st.columns(2)
        with a:
            plot(fig_score_hist(A, h=300), "me_hist")
        with b:
            if A["quiz_cols"]:
                q = df.groupby("cluster")[A["quiz_cols"]].mean().reindex(TIERS).reset_index().melt(
                    id_vars="cluster", var_name="Quiz", value_name="Mean score").dropna()
                fig = px.line(q, x="Quiz", y="Mean score", color="cluster", markers=True,
                              color_discrete_map=TIER_COLORS, category_orders={"cluster": TIERS},
                              labels={"cluster": "Tier"}) if len(q) else None
                plot(style_fig(fig, 300, "Mean quiz score by tier", legend=True) if fig else None, "me_quiz")
            else:
                prof = df.groupby("cluster")[A["features"]].mean().reindex(TIERS).T
                prof.index = [label(x) for x in prof.index]
                long = prof.reset_index().melt(id_vars="index", var_name="Tier", value_name="Mean")
                fig = px.bar(long, x="index", y="Mean", color="Tier", barmode="group",
                             color_discrete_map=TIER_COLORS, labels={"index": "Indicator"})
                fig.update_xaxes(tickangle=-35)
                plot(style_fig(fig, 300, "Average indicator by tier", legend=True), "me_prof")
    with t3:
        if "actual_class" in df.columns and df["actual_class"].notna().any():
            ct = pd.crosstab(df["cluster"], df["actual_class"]).reindex(TIERS)
            ct = ct[[c for c in ["High", "Medium", "Low"] if c in ct.columns]]
            a, b, c = st.columns([1, 1, 1.6])
            with a:
                metric_card("Adjusted Rand index", f"{m.get('ari_vs_actual_class', float('nan')):.3f}",
                            "0 = chance-level agreement, 1 = identical grouping.")
            with b:
                metric_card("Normalised mutual information", f"{m.get('nmi_vs_actual_class', float('nan')):.3f}",
                            "0 = no shared information, 1 = identical grouping.")
            with c:
                fig = px.imshow(ct.values, x=list(ct.columns), y=list(ct.index), text_auto=True, aspect="auto",
                                color_continuous_scale=HEAT)
                fig.update_layout(xaxis_title="Class label in the file", yaxis_title="Assigned tier")
                plot(style_fig(fig, 280, "Provided class label vs assigned tier"), "me_ct")
            st.caption("The file's Class label is not used to build the clusters; agreement is not prediction accuracy.")
        else:
            callout("note", "This dataset has no recognised class-label column (H / M / L), so there is nothing to "
                            "compare against.")


# ----------------------------------------------------------------------------
# Page: Indicator summary
# ----------------------------------------------------------------------------
def page_indicators():
    A = require_data("Indicator Summary", "Descriptive comparison of the available indicators across groups.")
    if A is None:
        return
    df, sch = A["df"], SCHEMAS[A["kind"]]
    st.caption("Descriptive summaries of columns in your file; not formally measured Course Outcome attainment.")
    group = st.radio("Compare across", ["Risk tier", "Grade"], horizontal=True, key="is_group")
    by, order, cmap = ("cluster", TIERS, TIER_COLORS) if group == "Risk tier" else ("Grade", GRADES, GRADE_COLORS)
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
        plot(style_fig(fig, 360, f"Average of each indicator by {group.lower()} (processed values)", legend=True),
             "is_bar")
    with b:
        mm = df[A["features"]]
        rng = (mm.max() - mm.min()).replace(0, np.nan)
        norm = ((mm - mm.min()) / rng).fillna(0).groupby(df[by]).mean().reindex(present)
        norm.columns = [label(c) for c in norm.columns]
        fig = px.imshow(norm.T.values, x=list(norm.index), y=list(norm.columns), text_auto=".2f", aspect="auto",
                        color_continuous_scale=HEAT, zmin=0, zmax=1)
        plot(style_fig(fig, 360, "Normalised comparison (0-1, for comparison only)"), "is_heat")

    with st.expander("How each indicator was prepared (raw, processed, normalised, derived)"):
        show_df(trail_table(A), height=fit_h(len(A["trail"]), 330))
        st.markdown(
            "- **Raw:** exactly as uploaded (Student Details and the 'original' export).\n"
            "- **Processed:** raw values after clipping to the 1st-99th percentile and median imputation; used in "
            "most charts and tables.\n"
            "- **Standardised:** processed values rescaled for K-Means; internal only.\n"
            "- **Normalised 0-1:** processed values rescaled across the class; used for the tier ranking and the "
            "comparison heatmaps.\n"
            f"- **Derived:** the score and Grade. {sch['score_note']}")


# ----------------------------------------------------------------------------
# Page: Student details
# ----------------------------------------------------------------------------
def page_student():
    A = require_data("Student Details", "One student's indicators alongside the class, with follow-up considerations.")
    if A is None:
        return
    df, orig, idc = A["df"], A["orig"], A["id_col"]
    dup_ids = df[idc].duplicated(keep=False)
    options = list(range(len(df)))
    fmt_fn = lambda i: f"{df[idc].iat[i]}" + (f" (row {i + 1})" if dup_ids.iat[i] else "")
    i = st.selectbox("Select a student (type to search)", options, format_func=fmt_fn, key="sd_pick")
    r = df.iloc[i]

    kpis([
        ("Student ID", str(r[idc]), "", NAVY),
        ("Grade", r["Grade"], "derived from score", GRADE_COLORS.get(r["Grade"], MUTED)),
        ("Risk tier", r["cluster"], f"risk level: {r['risk_level']}", TIER_COLORS.get(r["cluster"], MUTED)),
        (A["score_label"], f"{r[A['score_col']]:.1f}", f"class mean {df[A['score_col']].mean():.1f}", SAGE_D),
        ("Intervention list", "Yes" if r["In intervention list"] else "No", r["Flag basis"] or "not flagged",
         TERRA if r["In intervention list"] else SAGE_D),
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
        show_df(pd.DataFrame(rows), height=fit_h(len(rows), 400))
        st.caption("'Original' is the value in your file; 'Processed' is after clipping and median filling, as used in "
                   "the analysis. They differ only where a value was missing or extreme.")
    with b:
        if pcts:
            d = pd.DataFrame(pcts, columns=["Indicator", "Percentile"])
            fig = px.bar(d, x="Percentile", y="Indicator", orientation="h", range_x=[0, 100],
                         color_discrete_sequence=[SAGE_D], text=d["Percentile"].round(0))
            fig.add_vline(x=50, line_dash="dash", line_color=MUTED, annotation_text="Class median")
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Percentile within this class")
            plot(style_fig(fig, 350, "Where this student sits in the class"), "sd_pct")
        else:
            plot(None)

    rc = r["_reason_cols"]
    if rc:
        items = "".join(f"<li><b>{esc(label(c))}:</b> {esc(CONSIDERATIONS.get(c, CONSIDERATIONS['_engagement']))}</li>"
                        for c in rc)
        body = "<p>Indicators below the class lower quartile (or absence flag):</p><ul>" + items + "</ul>"
    elif r["In intervention list"]:
        body = (f"<p>Flagged by {esc(r['Flag basis'])}, but no single indicator is below the class lower quartile. "
                "Review the full record before deciding on any action.</p>")
    else:
        body = "<p>Not on the intervention list and no indicator is below the class lower quartile.</p>"
    card("Possible intervention considerations", body)
    st.caption("Prompts based on this student's recorded indicators relative to the class. They do not explain why a "
               "value is low and do not predict outcomes; use your own knowledge of the student.")


# ----------------------------------------------------------------------------
# Page: Export
# ----------------------------------------------------------------------------
def page_export():
    A = require_data("Export Reports", "Download results as CSV files for Excel or sharing.")
    if A is None:
        return
    st.caption("Original values are exactly as uploaded; processed values were clipped to the 1st-99th percentile and "
               "median-filled. Every student file has a Value_Basis column saying which one it holds. Derived score, "
               "Grade and tiers are always calculated from processed values.")
    choice = st.radio("Values for student columns",
                      ["Original uploaded values", "Processed values (used for analysis)"], horizontal=True,
                      key="ex_basis")
    basis = "original" if choice.startswith("Original") else "processed"
    stamp = datetime.now().strftime("%Y%m%d")
    role = st.session_state.get("role", "")
    denied = f"<p>Not available for the {esc(role)} role.</p>"

    c1, c2, c3 = st.columns(3)
    preview = None
    with c1:
        if can_export("results"):
            res = results_table(A, basis)
            preview = res
            card("Student results", f"<p>{len(res):,} students · {res.shape[1]} columns</p>")
            st.download_button("Download student results (CSV)", to_csv_bytes(res),
                               f"student_results_{basis}_{stamp}.csv", "text/csv", use_container_width=True,
                               key="dl_res")
        else:
            card("Student results", denied)
    with c2:
        if can_export("intervention"):
            flagged = results_table(A, basis, only_flagged=True)
            preview = flagged if preview is None else preview
            card("Intervention list", f"<p>{len(flagged):,} students (At-Risk tier or Grade F)</p>")
            st.download_button("Download intervention list (CSV)", to_csv_bytes(flagged),
                               f"intervention_list_{basis}_{stamp}.csv", "text/csv", use_container_width=True,
                               key="dl_flag", disabled=flagged.empty)
        else:
            card("Intervention list", denied)
    with c3:
        if can_export("summary"):
            summ = class_summary(A)
            card("Class summary", f"<p>{len(summ)} rows · counts and quality metrics</p>")
            st.download_button("Download class summary (CSV)", to_csv_bytes(summ), f"class_summary_{stamp}.csv",
                               "text/csv", use_container_width=True, key="dl_sum")
        else:
            card("Class summary", denied)
    if preview is not None:
        with st.expander("Preview of the first rows"):
            show_df(preview.head(15), height=fit_h(min(len(preview), 15), 330))
    st.caption("Quiz columns (if present) are never altered by the pipeline and are identical in both versions. "
               "Exports support faculty decisions and do not predict outcomes.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
ROUTES = {
    "Data Validation": page_validation, "Class Overview": page_overview,
    "Performance": page_performance, "At-Risk Students": page_risk, "Cluster Analysis": page_cluster,
    "Model Evaluation": page_model, "Indicator Summary": page_indicators, "Student Details": page_student,
    "Export Reports": page_export,
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

    # Everything below is reachable only after a successful sign-in.
    allowed = allowed_pages()
    if st.session_state.get("page") not in allowed:
        st.session_state["page"] = allowed[0] if allowed else DEFAULT_PAGE
    sidebar()                      # may start an analysis and change the page
    if not allowed:
        st.error("Your role does not have access to any page.")
        return
    navbar(allowed)
    ROUTES[st.session_state["page"]]()


if __name__ == "__main__":
    main()
