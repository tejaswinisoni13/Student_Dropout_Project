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
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide"
)

PRIMARY = "#6366F1"
PRIMARY_DARK = "#4F46E5"
NAVY = "#0F172A"

GRADE_COLORS = {
    "O": "#7c3aed",
    "A": "#16a34a",
    "B": "#2563eb",
    "F": "#dc2626"
}

TIERS = ["High-Performing", "Moderate", "At-Risk"]

# ─────────────────────────────────────────────
# MODERN SAAS DESIGN
# ─────────────────────────────────────────────

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: Inter, sans-serif;
}}

.stApp {{
    background: #f6f8fc;
}}

.main .block-container {{
    max-width: 1500px;
    padding: 1.4rem 2rem 2rem;
}}

[data-testid="stSidebar"] {{
    background: #ffffff;
    border-right: 1px solid #e8edf4;
}}

.dash-header {{
    background: linear-gradient(115deg, #111c36, #263c65);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin: .2rem 0 1rem;
    box-shadow: 0 10px 28px #0f172a18;
}}

.dash-title {{
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -.04em;
}}

.dash-subtitle {{
    color: #cbd5e1;
    font-size: .82rem;
    margin-top: .35rem;
}}

.schema-badge {{
    display: inline-block;
    background: #0d948822;
    color: #99f6e4;
    border: 1px solid #5eead455;
    border-radius: 30px;
    padding: .25rem .7rem;
    font-size: .7rem;
    font-weight: 700;
    margin-top: .65rem;
}}

.section-label {{
    color: #334155;
    font-size: .76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin: .8rem 0 .55rem;
    border-left: 3px solid {PRIMARY};
    padding-left: .65rem;
}}

.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: .85rem;
    margin: .8rem 0 1rem;
}}

.kpi-card {{
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 15px;
    padding: 1rem .85rem;
    box-shadow: 0 3px 12px #0f172a08;
}}

.kpi-val {{
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
}}

.kpi-lbl {{
    color: #64748b;
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: .4rem;
    font-weight: 700;
}}

.accent-blue .kpi-val {{ color: #6366f1; }}
.accent-green .kpi-val {{ color: #059669; }}
.accent-red .kpi-val {{ color: #e11d48; }}
.accent-amber .kpi-val {{ color: #d97706; }}
.accent-slate .kpi-val {{ color: #475569; }}

.note-box {{
    background: #ecfdf8;
    border: 1px solid #b8f1df;
    color: #285b50;
    border-radius: 12px;
    padding: .8rem 1rem;
    font-size: .8rem;
    margin: .5rem 0 1rem;
}}

.warn-box {{
    background: #fff8eb;
    border: 1px solid #fde2b4;
    color: #875b16;
    border-radius: 12px;
    padding: .8rem 1rem;
    font-size: .8rem;
    margin: .5rem 0 1rem;
}}

.info-box, .co-card {{
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: .65rem;
    box-shadow: 0 2px 8px #0f172a06;
}}

.info-box h4 {{
    color: #1e293b;
    font-size: .85rem;
    margin: 0 0 .35rem;
    font-weight: 700;
}}

.info-box p {{
    color: #64748b;
    font-size: .78rem;
    margin: 0;
    line-height: 1.55;
}}

.co-card {{
    text-align: center;
}}

.co-val {{
    font-size: 1.4rem;
    font-weight: 800;
    color: {PRIMARY_DARK};
}}

.co-lbl {{
    color: #64748b;
    font-size: .65rem;
    margin-top: .3rem;
    text-transform: uppercase;
    font-weight: 700;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
}}

.stTabs [data-baseweb="tab"] {{
    padding: .65rem .9rem;
    color: #64748b;
    font-weight: 600;
}}

.stTabs [aria-selected="true"] {{
    color: {PRIMARY_DARK} !important;
    border-bottom: 2px solid {PRIMARY} !important;
}}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 10px;
    font-weight: 600;
}}

</style>
""",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def section(label):
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True
    )


def kpi(value, label, accent="accent-blue"):
    return (
        f'<div class="kpi-card {accent}">'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'</div>'
    )


def chart_style(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            size=11,
            color="#64748b",
            family="Inter"
        ),
        margin=dict(t=35, b=25, l=30, r=20),
        xaxis=dict(gridcolor="#edf1f6"),
        yaxis=dict(gridcolor="#edf1f6")
    )
    return fig


def relative_score(series):
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return 0.0

    return float((s.mean() - lo) / (hi - lo) * 100)


# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.markdown(
        """
        <div style="
            max-width:470px;
            margin:3.5rem auto 1rem;
            background:#111c36;
            border-radius:18px;
            padding:2rem;
            text-align:center;
        ">
            <h2 style="color:white;margin:0">
                Faculty Portal Login
            </h2>
            <p style="color:#cbd5e1;font-size:.82rem">
                Student Performance Analytics & Early Intervention System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, login_col, _ = st.columns([1, 2, 1])

    with login_col:

        role = st.selectbox(
            "Select Role",
            [
                "Faculty Member / Professor",
                "System Administrator",
                "Course Coordinator"
            ]
        )

        user_id = st.text_input(
            "Faculty ID / Username",
            value="Prof. Sharma"
        )

        password = st.text_input(
            "Access Password",
            type="password",
            value="demo123"
        )

        if st.button("Login", use_container_width=True):

            if password.strip():

                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.user_id = user_id

                st.rerun()

            else:
                st.error("Please enter a password.")

    st.caption("Demo login interface. Enter any non-empty password.")

    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:

    st.markdown("## Faculty Portal")

    st.caption(
        f"{st.session_state.get('role', 'Faculty')} · "
        f"{st.session_state.get('user_id', '')}"
    )

    if st.button("Logout", use_container_width=True):

        st.session_state.authenticated = False
        st.rerun()

    st.divider()

    uploaded = st.file_uploader(
        "Upload class dataset (CSV)",
        type=["csv"]
    )

    st.caption(
        "Supported: M-Tech academic records and "
        "xAPI classroom engagement data."
    )

    st.caption(
        "Designed for cohorts of 50–500+ records. "
        "Actual minimum depends on the clustering algorithm."
    )


if uploaded is None:

    st.markdown(
        """
        <div style="
            background:white;
            border:2px dashed #cbd5e1;
            border-radius:18px;
            padding:3rem;
            text-align:center;
            margin-top:2rem;
        ">
            <h2>Upload a student dataset to begin</h2>
            <p style="color:#64748b">
                Choose a supported CSV file using the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ─────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────

try:

    raw = pd.read_csv(uploaded)

    if len(raw) < 4:
        st.error("Please upload at least 4 student records.")
        st.stop()

    prep = preprocess(raw)

    df, model, metrics = run_kmeans(
        prep,
        N_CLUSTERS
    )

    df = add_derived_grade(
        df,
        "primary_score"
    )

except Exception as exc:

    st.error(f"Dataset processing failed: {exc}")

    st.info(
        "Check that the CSV matches one of the supported dataset formats."
    )

    st.stop()


kind = prep["dataset_type"]
features = prep["available_cols"]
score_col = "primary_score"
score_label = prep["score_label"]
id_col = prep["id_col"]
quiz_cols = prep["quiz_cols"]

n = len(df)

counts = (
    df["cluster"]
    .value_counts()
    .reindex(TIERS)
    .fillna(0)
    .astype(int)
)

hp, moderate, risk = [
    int(counts[t]) for t in TIERS
]

avg = float(df[score_col].mean())

f_rate = float(
    (df["Grade"] == "F").mean() * 100
)


# ─────────────────────────────────────────────
# DASHBOARD HEADER
# ─────────────────────────────────────────────

dataset_name = (
    "Academic performance dataset"
    if kind == "mtech"
    else "Classroom engagement dataset"
)

st.markdown(
    f"""
    <div class="dash-header">

        <div class="dash-title">
            Student Performance Analytics
        </div>

        <div class="dash-subtitle">
            Faculty dashboard · {n} students evaluated ·
            {len(features)} indicators analyzed
        </div>

        <div class="schema-badge">
            {dataset_name}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
        <b>Faculty guidance:</b>
        Risk tiers compare students within the uploaded cohort.
        Use these results to identify students who may need support;
        review individual indicators before taking action.
    </div>
    """,
    unsafe_allow_html=True
)


tabs = st.tabs([
    "Home",
    "Overview",
    "Learning Curve",
    "Performance Analysis",
    "Cluster Analysis",
    "Indicator Summary"
])


# ════════════════════════════════════════════
# TAB 1 — HOME
# ════════════════════════════════════════════

with tabs[0]:

    st.markdown(
        '<div class="kpi-row">'
        + kpi(n, "Total Students", "accent-slate")
        + kpi(hp, "High-Performing", "accent-green")
        + kpi(moderate, "Moderate", "accent-amber")
        + kpi(risk, "At-Risk", "accent-red")
        + kpi(f"{avg:.1f}", score_label, "accent-blue")
        + '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns([3, 2])

    with a:

        section("Risk-tier distribution")

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=.48,
            color=counts.index,
            color_discrete_map=CLUSTER_COLORS
        )

        fig.update_traces(
            textinfo="label+percent+value"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with b:

        section("How to read this dashboard")

        st.markdown(
            """
            <div class="info-box">
                <h4>Automated processing</h4>
                <p>
                    Numeric validation, outlier handling,
                    median imputation, feature scaling
                    and K-Means clustering.
                </p>
            </div>

            <div class="info-box">
                <h4>Relative risk tiers</h4>
                <p>
                    Clusters are ranked by their average
                    normalized feature profile. They are not
                    fixed grade bands or top-quartile categories.
                </p>
            </div>

            <div class="info-box">
                <h4>Faculty action</h4>
                <p>
                    Review students flagged At-Risk or with
                    Grade F, then check attendance, quiz attempts
                    and other available indicators.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════
# TAB 2 — OVERVIEW
# ════════════════════════════════════════════

with tabs[1]:

    attendance_col = next(
        (
            c for c in ["Attendance", "attendance"]
            if c in df.columns
        ),
        None
    )

    attendance = (
        float(df[attendance_col].mean())
        if attendance_col
        else np.nan
    )

    st.markdown(
        '<div class="kpi-row">'
        + kpi(
            f"{risk/n*100:.1f}%",
            "At-Risk share",
            "accent-red"
        )
        + kpi(
            f"{attendance:.1f}" if pd.notna(attendance) else "—",
            "Average attendance",
            "accent-amber"
        )
        + kpi(
            f"{100-f_rate:.1f}%",
            "Non-F grade share",
            "accent-green"
        )
        + '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns([2, 3])

    with a:

        section("Performance score by tier")

        fig = px.box(
            df,
            x="cluster",
            y=score_col,
            color="cluster",
            points="all",
            color_discrete_map=CLUSTER_COLORS
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Risk tier",
            yaxis_title=score_label
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with b:

        section("Students for faculty follow-up")

        flagged = df[
            (df["risk_level"] == "High")
            | (df["Grade"] == "F")
        ].copy()

        if len(flagged):

            reason_cols = [
                c for c in [
                    "Attendance",
                    "QUIZ Attempt",
                    "Quiz Attempts",
                    "No. of Quiz Attempts",
                    "raisedhands",
                    "VisITedResources"
                ]
                if c in flagged.columns
            ]

            extra = [
                c for c in features
                if c not in reason_cols and c != score_col
            ][:2]

            cols = list(dict.fromkeys(
                [id_col, score_col]
                + reason_cols
                + extra
                + ["Grade", "cluster", "risk_level"]
            ))

            cols = [
                c for c in cols
                if c in flagged.columns
            ]

            view = flagged[cols].rename(
                columns={
                    id_col: "Student ID",
                    score_col: score_label,
                    "risk_level": "Risk Level",
                    "cluster": "Tier"
                }
            )

            query = st.text_input(
                "Search student ID",
                placeholder="Type an ID to filter"
            )

            if query:
                view = view[
                    view["Student ID"]
                    .astype(str)
                    .str.contains(
                        query,
                        case=False,
                        na=False
                    )
                ]

            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                height=280
            )

            st.caption(
                f"{len(flagged)} students flagged by At-Risk "
                "tier or Grade F. Categories may overlap."
            )

        else:

            st.success("No students are currently flagged.")


# ════════════════════════════════════════════
# TAB 3 — LEARNING CURVE
# ════════════════════════════════════════════

with tabs[2]:

    a, b = st.columns([1, 2])

    with a:

        section("Performance score distribution")

        fig = px.histogram(
            df,
            x=score_col,
            color="cluster",
            color_discrete_map=CLUSTER_COLORS,
            bins=16,
            barmode="overlay",
            opacity=.75
        )

        fig.update_layout(
            xaxis_title=score_label,
            yaxis_title="Students"
        )

        st.plotly_chart(
            chart_style(fig, 330),
            use_container_width=True
        )

    with b:

        if quiz_cols:

            section("Quiz performance by tier")

            rows = []

            for tier in TIERS:

                group = df[df["cluster"] == tier]

                for q in quiz_cols:

                    rows.append({
                        "Tier": tier,
                        "Quiz": q,
                        "Mean score": group[q].mean()
                    })

            trend = pd.DataFrame(rows)

            fig = px.line(
                trend,
                x="Quiz",
                y="Mean score",
                color="Tier",
                markers=True,
                color_discrete_map=CLUSTER_COLORS
            )

            st.plotly_chart(
                chart_style(fig, 330),
                use_container_width=True
            )

        else:

            section("Feature profile by tier")

            prof = (
                df.groupby("cluster")[features]
                .mean()
                .reindex(TIERS)
                .T
                .reset_index()
                .melt(
                    id_vars="index",
                    var_name="Tier",
                    value_name="Mean"
                )
            )

            fig = px.bar(
                prof,
                x="index",
                y="Mean",
                color="Tier",
                barmode="group",
                color_discrete_map=CLUSTER_COLORS
            )

            fig.update_layout(
                xaxis_title="Indicator"
            )

            st.plotly_chart(
                chart_style(fig, 330),
                use_container_width=True
            )

    st.markdown(
        """
        <div class="warn-box">
            Grade bands are derived from the primary score:
            O ≥ 75, A = 60–74.99, B = 50–59.99, F &lt; 50.
            Grade and cluster are separate labels; a Grade F
            student may be in another tier.
        </div>
        """,
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════
# TAB 4 — PERFORMANCE ANALYSIS
# ════════════════════════════════════════════

with tabs[3]:

    a, b = st.columns(2)

    with a:

        section("Grade distribution")

        gc = (
            df["Grade"]
            .value_counts()
            .reindex(["O", "A", "B", "F"])
            .fillna(0)
        )

        fig = go.Figure(
            go.Pie(
                labels=gc.index,
                values=gc.values,
                hole=.45,
                marker=dict(
                    colors=[
                        GRADE_COLORS[x]
                        for x in gc.index
                    ]
                ),
                textinfo="label+percent+value"
            )
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with b:

        section("Grades across risk tiers")

        matrix = (
            df.groupby(
                ["cluster", "Grade"],
                observed=False
            )
            .size()
            .reset_index(name="Students")
        )

        fig = px.bar(
            matrix,
            x="cluster",
            y="Students",
            color="Grade",
            barmode="group",
            color_discrete_map=GRADE_COLORS
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    section("Inspect an indicator")

    feature = st.selectbox(
        "Choose an indicator",
        features
    )

    fig = px.box(
        df,
        x="cluster",
        y=feature,
        color="cluster",
        points="all",
        color_discrete_map=CLUSTER_COLORS
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        chart_style(fig),
        use_container_width=True
    )


# ════════════════════════════════════════════
# TAB 5 — CLUSTER ANALYSIS
# ════════════════════════════════════════════

with tabs[4]:

    section("Clustering quality diagnostics")

    st.markdown(
        """
        <div class="warn-box">
            These metrics describe cluster separation, not predictive
            accuracy. A low Silhouette score or high Davies-Bouldin
            score indicates that groups may overlap. Interpret tiers
            cautiously.
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Silhouette score",
        f"{metrics['silhouette']:.3f}"
    )

    b.metric(
        "Davies-Bouldin index",
        f"{metrics['davies_bouldin']:.3f}"
    )

    c.metric(
        "Number of clusters",
        str(metrics["n_clusters"])
    )

    d.metric(
        "PCA variance (2D)",
        f"{prep['pca_variance'].sum()*100:.1f}%"
    )

    left, right = st.columns(2)

    with left:

        section("PCA cluster map")

        hover = [id_col, score_col, "Grade"]

        fig = px.scatter(
            df,
            x="pca_x",
            y="pca_y",
            color="cluster",
            color_discrete_map=CLUSTER_COLORS,
            hover_data=[
                x for x in hover
                if x in df.columns
            ],
            labels={
                "pca_x": "Principal component 1",
                "pca_y": "Principal component 2",
                "cluster": "Tier"
            }
        )

        fig.update_traces(
            marker=dict(
                size=9,
                line=dict(
                    width=.5,
                    color="white"
                )
            )
        )

        st.plotly_chart(
            chart_style(fig, 340),
            use_container_width=True
        )

        st.caption(
            f"The first two PCA components represent "
            f"{prep['pca_variance'].sum()*100:.1f}% of variance."
        )

    with right:

        section("Average feature profile by tier")

        profile = (
            df.groupby("cluster")[features]
            .mean()
            .reindex(TIERS)
            .T
        )

        st.dataframe(
            profile.style.format("{:.2f}"),
            use_container_width=True,
            height=340
        )

    st.markdown(
        """
        <div class="info-box">
            <h4>How tiers are assigned</h4>
            <p>
                K-Means forms three groups using scaled numeric indicators.
                Groups are then ranked by average normalized feature values.
                The highest-scoring group is labelled High-Performing and
                the lowest At-Risk. This is a relative cohort comparison,
                not a guaranteed prediction of dropout.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════
# TAB 6 — INDICATOR SUMMARY
# ════════════════════════════════════════════

with tabs[5]:

    st.markdown(
        """
        <div class="warn-box">
            <b>Important:</b> These are descriptive indicator summaries,
            not formally measured Course Outcome attainment values.
        </div>
        """,
        unsafe_allow_html=True
    )

    if kind == "mtech":

        definitions = {
            "Academic score": "AVG",
            "CGPA": "CGPA(%)",
            "Attendance": "Attendance",
            "Prior knowledge": "Prior Knowledge of subject"
        }

        indicators = {
            label: (
                float(df[col].mean())
                if col in df.columns
                else np.nan
            )
            for label, col in definitions.items()
        }

        engagement = [
            c for c in [
                "Resource Visited",
                "No. of Replies Posted",
                "No. of views on forum"
            ]
            if c in df.columns
        ]

        if engagement:

            indicators["Engagement"] = float(
                np.mean([
                    relative_score(df[c])
                    for c in engagement
                ])
            )

    else:

        definitions = {
            "Engagement score": score_col,
            "Class participation": "raisedhands",
            "Resource visits": "VisITedResources",
            "Discussion": "Discussion"
        }

        indicators = {
            label: (
                float(df[col].mean())
                if col in df.columns
                else np.nan
            )
            for label, col in definitions.items()
        }

        if "ParentAnsweringSurvey" in df.columns:

            indicators["Parent survey participation"] = float(
                df["ParentAnsweringSurvey"].mean() * 100
            )

    valid = {
        k: v
        for k, v in indicators.items()
        if pd.notna(v)
    }

    cards = st.columns(max(1, len(valid)))

    for (label, value), col in zip(valid.items(), cards):

        with col:

            st.markdown(
                f"""
                <div class="co-card">
                    <div class="co-val">{value:.1f}</div>
                    <div class="co-lbl">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if valid:

        section("Indicator comparison")

        summary = pd.DataFrame({
            "Indicator": list(valid.keys()),
            "Mean value": list(valid.values())
        })

        fig = px.bar(
            summary,
            x="Indicator",
            y="Mean value",
            text="Mean value"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
            marker_color=PRIMARY
        )

        st.plotly_chart(
            chart_style(fig, 340),
            use_container_width=True
        )


# ─────────────────────────────────────────────
# EXPORT CENTER
# ─────────────────────────────────────────────

st.divider()

section("Export center")

export_cols = list(dict.fromkeys(
    [id_col]
    + features
    + [
        score_col,
        "Grade",
        "cluster",
        "risk_level"
    ]
))

if "actual_class" in df.columns:
    export_cols.append("actual_class")

export_cols = [
    c for c in export_cols
    if c in df.columns
]

st.download_button(
    "Download processed student results (CSV)",
    data=df[export_cols].to_csv(index=False).encode("utf-8"),
    file_name="student_results_export.csv",
    mime="text/csv",
    use_container_width=True
)

st.caption(
    "Exported numeric values may include preprocessing adjustments "
    "such as winsorization and imputation. This dashboard is a "
    "decision-support tool, not a replacement for faculty judgement."
)