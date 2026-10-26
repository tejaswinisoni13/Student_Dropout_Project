"""
Streamlit Dashboard User Interface with Authentication Workflow,
Scalable Dataset Ingestion, and 6 Full Analytical Tabs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    CLUSTER_COLORS,
    N_CLUSTERS,
    preprocess,
    run_kmeans,
    add_derived_grade,
)

st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand palette (Unacademy-style: Deep Navy + Mint Green accent)
PRIMARY = "#00C896"
PRIMARY_DARK = "#00A278"
NAVY = "#0B1F3A"
NAVY_LIGHT = "#132A4D"

GRADE_COLORS = {"O": "#7c3aed", "A": "#16a34a", "B": "#2563eb", "F": "#dc2626"}
DATASET_LABELS = {"mtech": "Academic performance dataset", "xapi": "Classroom engagement dataset"}

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.main .block-container {{
    padding-top: .7rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}}

.dash-header {{
    background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 60%, {PRIMARY_DARK} 145%);
    border-top: 4px solid {PRIMARY};
    border-radius: 12px;
    padding: .9rem 1.6rem;
    margin-bottom: .6rem;
    box-shadow: 0 8px 18px -5px rgba(15,23,42,.25);
}}

.dash-title {{
    color: #ffffff !important;
    font-size: 1.3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -.02em;
}}

.dash-subtitle {{
    color: rgba(255,255,255,.75);
    font-size: .78rem;
    margin-top: .2rem;
}}

.schema-badge {{
    display: inline-block;
    background: rgba(0,200,150,.18);
    color: {PRIMARY};
    border: 1px solid rgba(0,200,150,.4);
    border-radius: 999px;
    padding: .15rem .6rem;
    font-size: .68rem;
    font-weight: 700;
    margin-top: .4rem;
}}

.section-label {{
    font-size: .76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin: .5rem 0 .4rem;
    border-left: 4px solid {PRIMARY};
    padding-left: .6rem;
    opacity: .9;
}}

.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: .6rem;
    margin-bottom: .6rem;
}}

.kpi-card {{
    background: var(--background-color);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 10px;
    padding: .65rem .7rem;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,.03);
    transition: transform .15s ease, box-shadow .15s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0,0,0,.08);
}}

.kpi-val {{ font-size: 1.45rem; font-weight: 800; line-height: 1.05; }}
.kpi-lbl {{ font-size: .62rem; opacity: .72; text-transform: uppercase; letter-spacing: .04em; margin-top: .2rem; font-weight: 700; }}

.accent-blue .kpi-val {{ color: {PRIMARY}; }}
.accent-green .kpi-val {{ color: #10b981; }}
.accent-red .kpi-val {{ color: #ef4444; }}
.accent-amber .kpi-val {{ color: #f59e0b; }}
.accent-slate .kpi-val {{ color: #64748b; }}

.note-box {{
    background: rgba(0,200,150,.07);
    border: 1px solid rgba(0,200,150,.28);
    border-radius: 8px;
    padding: .5rem .8rem;
    font-size: .78rem;
    margin-bottom: .5rem;
}}

.warn-box {{
    background: rgba(245,158,11,.08);
    border: 1px solid rgba(245,158,11,.28);
    border-radius: 8px;
    padding: .5rem .8rem;
    font-size: .78rem;
    margin-bottom: .5rem;
}}

.info-box {{
    background-color: rgba(128,128,128,.04);
    border: 1px solid rgba(128,128,128,.15);
    border-radius: 10px;
    padding: .6rem .9rem;
    margin-bottom: .4rem;
}}
.info-box h4 {{ font-size: .8rem; margin: 0 0 .2rem; font-weight: 700; }}
.info-box p {{ font-size: .77rem; margin: 0; line-height: 1.45; opacity: .85; }}

.co-card {{
    background-color: rgba(128,128,128,.04);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 10px;
    padding: .7rem;
    text-align: center;
    transition: transform .15s ease;
}}
.co-card:hover {{ transform: translateY(-2px); }}
.co-val {{ font-size: 1.3rem; font-weight: 800; color: {PRIMARY}; }}
.co-lbl {{ font-size: .6rem; opacity: .8; margin-top: .2rem; text-transform: uppercase; letter-spacing: .02em; font-weight: 600; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ padding: .4rem .8rem; }}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom-color: {PRIMARY} !important;
}}

div[data-testid="stVerticalBlock"] > div {{ gap: .4rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig, h=300):
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(128,128,128,.18)"),
        yaxis=dict(gridcolor="rgba(128,128,128,.18)"),
        margin=dict(t=35, b=25, l=30, r=20),
    )
    return fig


def kpi(val, lbl, accent="accent-blue"):
    return f'<div class="kpi-card {accent}"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>'


def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def relative_to_100(series):
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return 0.0
    return float((s.mean() - lo) / (hi - lo) * 100)


# ─────────────────────────────────────────────────────────────
# AUTHENTICATION & ROLE SELECTION WORKFLOW
# ─────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        f"""
        <div style="max-width: 450px; margin: 4rem auto; background: rgba(128,128,128,.04); 
                    border: 1px solid rgba(128,128,128,.2); border-radius: 14px; padding: 2.5rem; text-align: center;">
            <h2 style="color: {NAVY}; font-weight: 800; margin-bottom: 0.2rem;">Faculty Portal Login</h2>
            <p style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 1.5rem;">Student Performance Analytics & Early Intervention System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        role = st.selectbox("Select Role", ["Faculty Member / Professor", "System Administrator", "Course Coordinator"])
        user_id = st.text_input("Faculty ID / Username", value="Prof. Sharma")
        passkey = st.text_input("Access Password", type="password", value="demo123")
        
        if st.button("Secure Login", use_container_width=True):
            if passkey.strip() != "":
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.user_id = user_id
                st.rerun()
            else:
                st.error("Please enter a valid password.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# SIDEBAR & DATA INGESTION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### Portal: {st.session_state.role}")
    st.markdown(f"<p style='font-size:0.75rem;opacity:0.8;'>Logged in as: <strong>{st.session_state.user_id}</strong></p>", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()
    uploaded = st.file_uploader("Upload Class CSV (Supports 50 to 500+ students)", type=["csv"])
    st.divider()
    st.markdown(
        """
        <p style="font-size:.76rem;opacity:.75;line-height:1.5;">
        Automatically groups cohorts (100–200+ records) into three relative risk tiers using K-Means ($K=3$)[cite: 9, 10, 11].<br><br>
        <strong>Supported Schemas:</strong><br>
        - Academic records (M-Tech dataset)[cite: 9, 10, 11]<br>
        - Classroom engagement logs (xAPI dataset)[cite: 9, 10, 11]
        </p>
        """,
        unsafe_allow_html=True,
    )

if not uploaded:
    st.markdown(
        """
        <div style="background-color:rgba(128,128,128,.04);border:2px dashed rgba(128,128,128,.22);
                    border-radius:14px;padding:3rem;text-align:center;margin-top:1.5rem;">
          <h2 style="margin:.4rem 0;font-weight:700;">Upload Student Dataset to Initialize Dashboard</h2>
          <p style="opacity:.7;font-size:.83rem;">Upload any CSV matching academic records or engagement logs to begin analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

try:
    prep = preprocess(df_raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

dataset_type = prep["dataset_type"]
score_col = "primary_score"
score_label = prep["score_label"]
id_col = prep["id_col"]
features = prep["available_cols"]
quiz_cols = prep["quiz_cols"]

df, kmeans, metrics = run_kmeans(prep, N_CLUSTERS)
df = add_derived_grade(df, score_col)

n_students = len(df)
hp_count = int((df["cluster"] == "High-Performing").sum())
moderate_count = int((df["cluster"] == "Moderate").sum())
risk_count = int((df["cluster"] == "At-Risk").sum())

avg_perf = float(df[score_col].mean())
f_grade_rate = float((df["Grade"] == "F").mean() * 100)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="dash-header">
        <div class="dash-title">Student Performance Analytics Dashboard</div>
        <div class="dash-subtitle">
            Automated cohort grouping • {n_students} students evaluated • {len(features)} performance indicators tracked
        </div>
        <div class="schema-badge">Detected Schema: {DATASET_LABELS.get(dataset_type, dataset_type)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note-box">
    <strong>Faculty Guidance:</strong> Students are clustered into relative tiers — 
    <strong>High-Performing</strong>, <strong>Moderate</strong>, and <strong>At-Risk</strong> — 
    comparing peer performance within this cohort. Use the At-Risk roster for early intervention outreach.
    </div>
    """,
    unsafe_allow_html=True,
)

# 6 Tabs requested in PRD & specifications
tabs = st.tabs(["Home", "Overview", "Learning Curve", "Performance Analysis", "Cluster Analysis", "Course Outcomes"])


# ════════════════════════════════════════════════════════════
# TAB 0: HOME
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        f"""
        <div class="kpi-row">
          {kpi(n_students, "Total Students", "accent-slate")}
          {kpi(hp_count, "High-Performing", "accent-green")}
          {kpi(moderate_count, "Moderate", "accent-amber")}
          {kpi(risk_count, "At-Risk", "accent-red")}
          {kpi(f"{avg_perf:.1f}", score_label, "accent-blue")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2])

    with c1:
        section("Risk-Tier Distribution")
        counts = (
            df["cluster"].value_counts()
            .reindex(["High-Performing", "Moderate", "At-Risk"])
            .fillna(0).reset_index()
        )
        counts.columns = ["Cluster", "Count"]
        fig = px.pie(counts, names="Cluster", values="Count", color="Cluster",
                     color_discrete_map=CLUSTER_COLORS, hole=.45)
        fig.update_traces(textinfo="label+percent+value", textfont_size=12)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    with c2:
        section("Dashboard Guide & System Architecture")
        st.markdown(
            f"""
            <div class="info-box">
              <h4>Automated Pipeline</h4>
              <p>Winsorization (1st/99th pct), median imputation, and StandardScaler applied before K-Means ($K=3$).</p>
            </div>
            <div class="info-box">
              <h4>Risk Categorization</h4>
              <p><strong>High-Performing</strong> = Top quartile profile[cite: 11, 12]. 
              <strong>Moderate</strong> = Peer median. <strong>At-Risk</strong> = Requires immediate faculty attention[cite: 11, 12].</p>
            </div>
            <div class="info-box">
              <h4>Cohort Scale</h4>
              <p>Optimized for classes ranging from 50 to 500+ students seamlessly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════
with tabs[1]:
    avg_attendance = df["Attendance"].mean() if "Attendance" in df.columns else np.nan

    st.markdown(
        f"""
        <div class="kpi-row">
          {kpi(f"{risk_count/n_students*100:.1f}%", "At-Risk Rate", "accent-red")}
          {kpi(f"{avg_attendance:.1f}" if not np.isnan(avg_attendance) else "-", "Avg Attendance Score", "accent-amber")}
          {kpi(f"{100-f_grade_rate:.1f}%", "Non-F Grade Share", "accent-green")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 3])

    with c1:
        section(f"{score_label} Spread by Risk Tier")
        fig = px.box(df, x="cluster", y=score_col, color="cluster",
                     color_discrete_map=CLUSTER_COLORS, points="all")
        fig.update_layout(showlegend=False, xaxis_title="Risk Tier", yaxis_title=score_label)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    with c2:
        section("At-Risk Student Roster (Early Intervention)")
        at_risk = df[df["risk_level"] == "High"].copy()

        if len(at_risk):
            extra_feats = [c for c in features if c != score_col][:3]
            show_cols = ([id_col] if id_col else []) + [score_col] + extra_feats + ["Grade", "risk_level"]
            show_cols = [c for c in show_cols if c in at_risk.columns]

            table = at_risk[show_cols].copy()
            rename = {id_col: "Student ID", score_col: score_label, "risk_level": "Risk Level"}
            table = table.rename(columns={k: v for k, v in rename.items() if k in table.columns})

            search = st.text_input("Search by Student ID", "", label_visibility="collapsed", placeholder="Search by Student ID")
            if search and "Student ID" in table.columns:
                table = table[table["Student ID"].astype(str).str.contains(search, case=False, na=False)]

            st.dataframe(table, use_container_width=True, hide_index=True, height=230)
            st.caption(f"Showing {len(table)} of {len(at_risk)} At-Risk students.")
        else:
            st.success("No students currently flagged in the At-Risk tier.")


# ════════════════════════════════════════════════════════════
# TAB 2: LEARNING CURVE
# ════════════════════════════════════════════════════════════
with tabs[2]:
    c1, c2 = st.columns([1, 2])

    with c1:
        section(f"{score_label} Distribution")
        fig = px.histogram(df, x=score_col, color="cluster", color_discrete_map=CLUSTER_COLORS,
                           nbins=16, barmode="overlay", opacity=.75)
        fig.update_layout(xaxis_title=score_label, yaxis_title="Students", showlegend=True,
                          legend=dict(orientation="h", y=-.25))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    with c2:
        if quiz_cols:
            section("Quiz Performance Trend (Q1 - Q6) by Risk Tier")
            rows = []
            for tier in ["High-Performing", "Moderate", "At-Risk"]:
                group = df[df["cluster"] == tier]
                if group.empty:
                    continue
                for quiz in quiz_cols:
                    rows.append({"Risk Tier": tier, "Quiz": quiz.strip(), "Mean Score": group[quiz].mean()})
            trend = pd.DataFrame(rows)
            fig = px.line(trend, x="Quiz", y="Mean Score", color="Risk Tier",
                          color_discrete_map=CLUSTER_COLORS, markers=True)
            fig.update_traces(line_width=2.5, marker_size=7)
            fig.update_layout(xaxis_title="Quiz Assessment", yaxis_title="Mean Quiz Score")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        else:
            section("Feature Profile by Risk Tier")
            profile_feats = [c for c in features if c != score_col][:6]
            if profile_feats:
                prof = df.groupby("cluster")[profile_feats].mean().reindex(["High-Performing", "Moderate", "At-Risk"])
                melted = prof.reset_index().melt(id_vars="cluster", var_name="Feature", value_name="Mean Value")
                fig = px.bar(melted, x="Feature", y="Mean Value", color="cluster", barmode="group",
                            color_discrete_map=CLUSTER_COLORS)
                fig.update_layout(xaxis_title="Feature", yaxis_title="Mean Value")
                st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    st.markdown(
        f"""
        <div class="note-box">
        Grade bands derive from {score_label}: O &ge; 75, A = 60-74.99, B = 50-59.99, F &lt; 50[cite: 11, 12].
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# TAB 3: PERFORMANCE ANALYSIS
# ════════════════════════════════════════════════════════════
with tabs[3]:
    c1, c2 = st.columns(2)

    with c1:
        section("Grade Letter Distribution")
        grade_counts = df["Grade"].value_counts().reindex(["O", "A", "B", "F"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=grade_counts.index.tolist(), values=grade_counts.values.tolist(), hole=.45,
            marker=dict(colors=[GRADE_COLORS[g] for g in grade_counts.index]),
            textinfo="label+percent+value",
        ))
        fig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=15, b=15, l=15, r=15), font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Grades per Risk Tier Matrix")
        gcls = df.groupby(["cluster", "Grade"], observed=False).size().reset_index(name="Count")
        fig = px.bar(gcls, x="cluster", y="Count", color="Grade", barmode="group",
                     text="Count", color_discrete_map=GRADE_COLORS)
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Risk Tier", yaxis_title="Number of Students")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    section("Deep Dive Feature Inspection")
    if features:
        selected_feature = st.selectbox("Select indicator for boxplot comparison", features, index=0)
        fig = px.box(df, x="cluster", y=selected_feature, color="cluster",
                     color_discrete_map=CLUSTER_COLORS, points="all")
        fig.update_layout(xaxis_title="Risk Tier", yaxis_title=selected_feature, showlegend=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4: CLUSTER ANALYSIS (K-Means & PCA Diagnostics)
# ════════════════════════════════════════════════════════════
with tabs[4]:
    section("K-Means Clustering Diagnostics & Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi(f"{metrics['silhouette']:.4f}", "Silhouette Score", "accent-green"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi(f"{metrics['davies_bouldin']:.4f}", "Davies-Bouldin Index", "accent-blue"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi(f"{metrics['n_clusters']}", "Clusters (K=3)", "accent-slate"), unsafe_allow_html=True)
    with col4:
        if "ari_vs_actual_class" in metrics:
            st.markdown(kpi(f"{metrics['ari_vs_actual_class']:.4f}", "ARI vs Ground Truth", "accent-amber"), unsafe_allow_html=True)
        else:
            st.markdown(kpi(f"{prep['pca_variance'].sum()*100:.1f}%", "PCA Variance (2D)", "accent-amber"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        section("PCA 2D Spatial Mapping")
        fig = px.scatter(
            df, x="pca_x", y="pca_y", color="cluster",
            color_discrete_map=CLUSTER_COLORS,
            hover_data=[id_col, score_col, "Grade"] if id_col else [score_col, "Grade"],
            labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2", "cluster": "Risk Tier"}
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        st.caption(f"PCA 2D captures {prep['pca_variance'].sum()*100:.1f}% of cumulative variance.")

    with c2:
        section("Cluster Mean Feature Profiles")
        profile_df = df.groupby("cluster")[features].mean().reindex(["High-Performing", "Moderate", "At-Risk"]).T
        st.dataframe(profile_df.style.format("{:.2f}"), use_container_width=True, height=310)

    section("Clustering Architecture & Interpretation")
    st.markdown(
        """
        <div class="info-box">
          <h4>Unsupervised Segmentation</h4>
          <p>Features are normalized via StandardScaler and partitioned using <strong>K-Means ($K=3$)</strong>. 
          Clusters are automatically ranked by normalized composite score to assign High-Performing, Moderate, and At-Risk tiers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# TAB 5: COURSE OUTCOMES
# ════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        """
        <div class="warn-box">
        <strong>Note:</strong> Descriptive indicators simulated from cohort parameters for institutional outcome mapping.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataset_type == "mtech":
        indicators = {
            "CO1: Academic Performance": float(df["AVG"].mean()) if "AVG" in df.columns else float(df[score_col].mean()),
            "CO2: CGPA Performance": float(df["CGPA(%)"].mean()) if "CGPA(%)" in df.columns else 0.0,
            "CO3: Attendance Indicator": relative_to_100(df["Attendance"]) if "Attendance" in df.columns else 0.0,
            "CO4: Prior Knowledge Indicator": (
                float(df["Prior Knowledge of subject"].mean() / 5 * 100)
                if "Prior Knowledge of subject" in df.columns else 0.0
            ),
            "CO5: Engagement Indicator": (
                np.mean([
                    relative_to_100(df["Resource Visited"]),
                    relative_to_100(df["No. of Replies Posted"]),
                    relative_to_100(df["No. of views on forum"]),
                ])
                if all(c in df.columns for c in ["Resource Visited", "No. of Replies Posted", "No. of views on forum"])
                else 0.0
            ),
        }
    else:
        indicators = {
            "CO1: Engagement Score": float(df[score_col].mean()),
            "CO2: Class Participation": relative_to_100(df["raisedhands"]) if "raisedhands" in df.columns else 0.0,
            "CO3: Resource Engagement": relative_to_100(df["VisITedResources"]) if "VisITedResources" in df.columns else 0.0,
            "CO4: Discussion Participation": relative_to_100(df["Discussion"]) if "Discussion" in df.columns else 0.0,
            "CO5: Parent Involvement": (
                np.mean([
                    float(df["ParentAnsweringSurvey"].mean() * 100),
                    float(df["ParentschoolSatisfaction"].mean() * 100),
                ])
                if all(c in df.columns for c in ["ParentAnsweringSurvey", "ParentschoolSatisfaction"])
                else 0.0
            ),
        }

    co = pd.DataFrame({
        "CO": list(indicators.keys()),
        "Indicator": [round(max(0, min(100, v)), 1) for v in indicators.values()],
    })

    cards = st.columns(len(co))
    for i, (_, row) in enumerate(co.iterrows()):
        with cards[i]:
            st.markdown(
                f'<div class="co-card"><div class="co-val">{row["Indicator"]}%</div>'
                f'<div class="co-lbl">{row["CO"]}</div></div>',
                unsafe_allow_html=True,
            )

    section("Attainment Bar Chart")
    fig = px.bar(co, x="CO", y="Indicator", text="Indicator", title=None)
    fig.update_traces(texttemplate="%{text}%", textposition="outside", marker_color=PRIMARY)
    fig.update_layout(yaxis=dict(range=[0, 115]), xaxis_title="Course Outcome Indicator", yaxis_title="Attainment (%)")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    best = co.loc[co["Indicator"].idxmax()]
    worst = co.loc[co["Indicator"].idxmin()]
    c1, c2 = st.columns(2)
    c1.success(f"Strongest indicator: {best['CO']} — {best['Indicator']}%")
    c2.warning(f"Needs attention: {worst['CO']} — {worst['Indicator']}%")


# ─────────────────────────────────────────────────────────────
# EXPORT CENTER
# ─────────────────────────────────────────────────────────────
st.divider()

export_cols = ([id_col] if id_col else []) + features + [score_col, "Grade", "cluster", "risk_level"]
if "actual_class" in df.columns:
    export_cols.append("actual_class")
export_cols = [c for c in export_cols if c in df.columns]

st.download_button(
    "Download Processed Student Results (CSV)",
    data=df[export_cols].to_csv(index=False),
    file_name="student_results_export.csv",
    mime="text/csv",
)

st.caption("Results exported include assigned risk tier, primary performance score, and letter grade per student.")
