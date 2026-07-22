import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

CLR = {
    "High-Performing": "#2563eb",
    "Moderate":        "#d97706",
    "At-Risk":         "#dc2626"
}

FEAT_COLS = [
    "10th Score(%)", "12th Score(%)", "CGPA(%)", "AVG",
    "QUIZ Attempt", "Attendance", "Aim/Objective",
    "Prior Knowledge of subject", "Resource Visited",
    "No. of Replies Posted", "No. of views on forum"
]

QUIZ_COLS = ["QUIZ 1","QUIZ 2","QUIZ 3","QUIZ 4","QUIZ 5","QUIZ 6"]

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"]          { background: #1e293b; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; }
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Header ── */
.dash-header {
    background: #1e293b;
    border-radius: 10px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.25rem;
}
.dash-title {
    font-size: 1.4rem; font-weight: 700;
    color: #f1f5f9; margin: 0; letter-spacing: .01em;
}
.dash-sub {
    font-size: .82rem; color: #94a3b8; margin-top: .3rem;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: .85rem; margin-bottom: 1.25rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kpi-val { font-size: 1.9rem; font-weight: 800; color: #1e293b; }
.kpi-lbl { font-size: .72rem; color: #64748b; text-transform: uppercase;
           letter-spacing: .05em; margin-top: .25rem; font-weight: 600; }
.kpi-card.accent-blue  .kpi-val { color: #2563eb; }
.kpi-card.accent-green .kpi-val { color: #16a34a; }
.kpi-card.accent-red   .kpi-val { color: #dc2626; }
.kpi-card.accent-amber .kpi-val { color: #d97706; }
.kpi-card.accent-slate .kpi-val { color: #475569; }

/* ── Section Label ── */
.section-label {
    font-size: .8rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: .06em;
    margin: 1.1rem 0 .6rem; border-left: 3px solid #2563eb;
    padding-left: .6rem;
}

/* ── Info Box ── */
.info-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.info-box h4 { color: #1e293b; font-size: .9rem; margin: 0 0 .5rem; font-weight: 700; }
.info-box p  { color: #475569; font-size: .85rem; margin: 0; line-height: 1.7; }

/* ── Comparison Table ── */
.ctbl { width: 100%; border-collapse: collapse; font-size: .85rem; }
.ctbl th {
    background: #f1f5f9; color: #64748b; font-weight: 700;
    padding: .75rem 1rem; border-bottom: 2px solid #e2e8f0;
    text-align: left; text-transform: uppercase;
    font-size: .75rem; letter-spacing: .04em;
}
.ctbl td {
    padding: .8rem 1rem; border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
}
.ctbl tr:hover td { background: #f8fafc; }
.best-tag {
    background: #dcfce7; color: #16a34a;
    padding: .15rem .5rem; border-radius: 4px;
    font-size: .7rem; font-weight: 700; margin-left: .3rem;
}

/* ── Metric Badges ── */
.mbadge { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: .85rem; }
.mb {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: .75rem 1rem;
    text-align: center; flex: 1; min-width: 120px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.mbv { font-size: 1.4rem; font-weight: 800; color: #1e293b; }
.mbl { font-size: .68rem; color: #64748b; margin-top: .15rem;
       text-transform: uppercase; letter-spacing: .04em; }

/* ── Student Table ── */
.stbl { width: 100%; border-collapse: collapse; font-size: .83rem; }
.stbl th {
    background: #f1f5f9; color: #64748b; font-weight: 700;
    padding: .65rem .9rem; border-bottom: 2px solid #e2e8f0;
    text-align: left; font-size: .73rem;
    text-transform: uppercase; letter-spacing: .04em;
}
.stbl td { padding: .7rem .9rem; border-bottom: 1px solid #f1f5f9; color: #1e293b; }
.stbl tr:hover td { background: #f8fafc; }
.badge-h { background:#fee2e2; color:#dc2626; padding:.2rem .55rem;
           border-radius:4px; font-weight:700; font-size:.73rem; }
.badge-m { background:#fef3c7; color:#d97706; padding:.2rem .55rem;
           border-radius:4px; font-weight:700; font-size:.73rem; }
.badge-l { background:#dcfce7; color:#16a34a; padding:.2rem .55rem;
           border-radius:4px; font-weight:700; font-size:.73rem; }

/* ── CO Card ── */
.co-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.co-val { font-size: 1.7rem; font-weight: 800; color: #2563eb; }
.co-lbl { font-size: .7rem; color: #64748b; margin-top: .2rem;
          text-transform: uppercase; letter-spacing: .04em; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### Dashboard Settings")
    uploaded     = st.file_uploader("Upload Student CSV", type=["csv"])
    st.divider()
    n_clusters   = st.slider("Number of Clusters", 2, 6, 3)
    algorithm    = st.selectbox("Clustering Algorithm",
                                ["K-Means","GMM","Hierarchical"])
    st.divider()
    st.markdown("**Filter Students**")
    risk_filter  = st.multiselect("Risk Level",
                                  ["High","Medium","Low"],
                                  default=["High","Medium","Low"])
    clust_filter = st.multiselect("Cluster Group",
                                  ["High-Performing","Moderate","At-Risk"],
                                  default=["High-Performing",
                                           "Moderate","At-Risk"])
    st.divider()
    st.markdown("""
    <p style="font-size:.78rem;color:#64748b;line-height:1.6;">
    This dashboard uses unsupervised machine learning to automatically
    group students based on academic performance and engagement metrics.
    </p>
    """, unsafe_allow_html=True)

# ── HELPERS ──
def dl(fig, h=480):
    fig.update_layout(
        height=h,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#1e293b", size=12),
        title_font=dict(color="#1e293b", size=14, family="sans-serif"),
        legend=dict(font=dict(color="#475569"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#e2e8f0",
                   tickfont=dict(color="#64748b"),
                   title_font=dict(color="#64748b")),
        yaxis=dict(gridcolor="#e2e8f0",
                   tickfont=dict(color="#64748b"),
                   title_font=dict(color="#64748b"))
    )
    return fig

def assign_names(df, lbl, name, feats, n):
    dn = df[feats].copy()
    for c in feats:
        r = dn[c].max() - dn[c].min()
        if r > 0: dn[c] = (dn[c] - dn[c].min()) / r
    sc = {l: dn.loc[df[lbl]==l].mean().mean() for l in range(n)}
    sl = sorted(sc, key=sc.get, reverse=True)
    nm = (["High-Performing","Moderate","At-Risk"] +
          [f"Cluster {i}" for i in range(4, n+1)])[:n]
    df[name] = df[lbl].map({sl[i]: nm[i] for i in range(n)})
    return df

def grade(avg):
    if   avg >= 75: return "O"
    elif avg >= 60: return "A"
    elif avg >= 50: return "B"
    else:           return "F"

def kpi(val, lbl, accent="accent-blue"):
    return (f'<div class="kpi-card {accent}">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>')

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>',
                unsafe_allow_html=True)

# ════════════════════════════════════════════
if uploaded:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip()
    if "Roll Number" in df_raw.columns:
        df_raw.rename(columns={"Roll Number":"Student ID"}, inplace=True)

    FEAT  = [c for c in FEAT_COLS if c in df_raw.columns]
    QUIZZ = [c for c in QUIZ_COLS  if c in df_raw.columns]
    df    = df_raw.copy()

    for col in FEAT:
        df[col] = df[col].clip(df[col].quantile(.01),
                               df[col].quantile(.99))
    df[FEAT] = SimpleImputer(strategy="mean").fit_transform(df[FEAT])
    X        = StandardScaler().fit_transform(df[FEAT])
    Xp       = PCA(n_components=2, random_state=42).fit_transform(X)
    df["px"], df["py"] = Xp[:,0], Xp[:,1]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["km_lbl"] = km.fit_predict(X)
    df = assign_names(df, "km_lbl", "km_cls", FEAT, n_clusters)

    gm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    gm.fit(X)
    df["gm_lbl"]  = gm.predict(X)
    df["gm_prob"] = gm.predict_proba(X).max(axis=1)
    df = assign_names(df, "gm_lbl", "gm_cls", FEAT, n_clusters)

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df["hc_lbl"] = hc.fit_predict(X)
    df = assign_names(df, "hc_lbl", "hc_cls", FEAT, n_clusters)

    amap = {"K-Means":"km_cls","GMM":"gm_cls","Hierarchical":"hc_cls"}
    df["cluster"]      = df[amap[algorithm]]
    df["dropout_risk"] = df["cluster"].map(
        {"High-Performing":"Low","Moderate":"Medium","At-Risk":"High"})
    df["Grade"] = df["AVG"].apply(grade)

    sil = {"K-Means":      round(silhouette_score(X, df["km_lbl"]),4),
           "GMM":          round(silhouette_score(X, df["gm_lbl"]),4),
           "Hierarchical": round(silhouette_score(X, df["hc_lbl"]),4)}
    dbs = {"K-Means":      round(davies_bouldin_score(X, df["km_lbl"]),4),
           "GMM":          round(davies_bouldin_score(X, df["gm_lbl"]),4),
           "Hierarchical": round(davies_bouldin_score(X, df["hc_lbl"]),4)}
    best = max(sil, key=sil.get)

    df_f = df[df["dropout_risk"].isin(risk_filter) &
              df["cluster"].isin(clust_filter)]

    N        = len(df_f)
    hp       = (df_f["cluster"]=="High-Performing").sum()
    mo       = (df_f["cluster"]=="Moderate").sum()
    ar       = (df_f["cluster"]=="At-Risk").sum()
    avg_perf = df_f[["CGPA(%)","AVG"]].mean().mean()
    pass_r   = (df_f["Grade"]!="F").sum() / max(N,1) * 100
    drop_r   = 100 - pass_r

    # ── HEADER ──
    st.markdown(f"""
    <div class="dash-header">
      <div class="dash-title">Student Performance Analytics Dashboard</div>
      <div class="dash-sub">
        {N} students loaded &nbsp;|&nbsp;
        Active algorithm: {algorithm} &nbsp;|&nbsp;
        Best performing: {best} (Silhouette = {sil[best]})
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "Home",
        "Overview",
        "Learning Curve",
        "Performance Analysis",
        "Cluster Analysis",
        "Course Outcomes"
    ])

    # ══════════════════════════════
    # HOME
    # ══════════════════════════════
    with t1:
        st.markdown(f"""
        <div class="kpi-row">
          {kpi(N, "Total Students", "accent-slate")}
          {kpi(hp, "High-Performing", "accent-green")}
          {kpi(mo, "Moderate", "accent-amber")}
          {kpi(ar, "At-Risk", "accent-red")}
          {kpi(f"{avg_perf:.1f}%", "Avg Performance", "accent-blue")}
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3,2])
        with c1:
            section("Cluster Distribution")
            sz  = df_f["cluster"].value_counts().reset_index()
            sz.columns = ["Cluster","Count"]
            fig = px.pie(sz, names="Cluster", values="Count",
                         color="Cluster", color_discrete_map=CLR,
                         hole=0.45,
                         title="Student Cluster Distribution")
            fig.update_traces(textfont_size=14,
                              textinfo="label+percent+value")
            st.plotly_chart(dl(fig, 460), use_container_width=True)

        with c2:
            section("Dashboard Guide")
            st.markdown("""
            <div class="info-box" style="margin-bottom:.75rem;">
              <h4>What is this dashboard?</h4>
              <p>This tool uses machine learning to automatically
              segment students into performance groups based on
              academic and engagement data — without manual grading.</p>
            </div>
            <div class="info-box" style="margin-bottom:.75rem;">
              <h4>How to use it</h4>
              <p>
              1. Upload the student CSV from the sidebar.<br>
              2. Select the clustering algorithm.<br>
              3. Use filters to focus on specific student groups.<br>
              4. Navigate tabs to explore different analyses.
              </p>
            </div>
            <div class="info-box">
              <h4>Cluster Groups Explained</h4>
              <p>
              <strong style="color:#2563eb">High-Performing</strong>
              — Strong academics and engagement.<br>
              <strong style="color:#d97706">Moderate</strong>
              — Average performance, may need support.<br>
              <strong style="color:#dc2626">At-Risk</strong>
              — Low scores, high dropout probability.
              </p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════
    # OVERVIEW
    # ══════════════════════════════
    with t2:
        st.markdown(f"""
        <div class="kpi-row">
          {kpi(f"{avg_perf:.1f}%", "Avg CGPA + AVG", "accent-blue")}
          {kpi(f"{ar/max(N,1)*100:.0f}%", "At-Risk Rate", "accent-red")}
          {kpi(f"{df_f['Attendance'].mean():.1f}", "Avg Attendance", "accent-amber")}
          {kpi(f"{pass_r:.0f}%", "Pass Rate", "accent-green")}
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            section("Dropout Risk Distribution")
            rc = (df_f["dropout_risk"]
                      .value_counts().reset_index()
                      .rename(columns={"dropout_risk":"Risk","count":"Count"}))
            fig = px.bar(rc, x="Risk", y="Count", color="Risk",
                         color_discrete_map={"High":"#dc2626",
                                             "Medium":"#d97706",
                                             "Low":"#16a34a"},
                         text="Count",
                         title="Number of Students by Risk Level")
            fig.update_traces(textposition="outside",
                              marker_line_color="white",
                              marker_line_width=1.5)
            fig.update_layout(showlegend=False)
            st.plotly_chart(dl(fig, 460), use_container_width=True)

        with c2:
            section("AVG Score by Cluster")
            fig = px.box(df_f, x="cluster", y="AVG",
                         color="cluster", color_discrete_map=CLR,
                         title="Distribution of AVG Scores per Cluster",
                         points="all")
            fig.update_layout(showlegend=False,
                              xaxis_title="Cluster",
                              yaxis_title="AVG Score")
            st.plotly_chart(dl(fig, 460), use_container_width=True)

        section("At-Risk Student List")
        at_risk = df_f[df_f["dropout_risk"]=="High"]
        if len(at_risk):
            cols_show = [c for c in
                         ["Student ID","CGPA(%)","AVG",
                          "Attendance","Grade","dropout_risk"]
                         if c in at_risk.columns]
            rows = ""
            for _, row in at_risk[cols_show].iterrows():
                sid  = row.get("Student ID","—")
                cgpa = row.get("CGPA(%)", 0)
                avg_ = row.get("AVG", 0)
                att  = row.get("Attendance", 0)
                gr   = row.get("Grade","—")
                rows += f"""<tr>
                  <td>{sid}</td>
                  <td>{cgpa:.1f}%</td>
                  <td>{avg_:.1f}</td>
                  <td>{att}</td>
                  <td>{gr}</td>
                  <td><span class="badge-h">High Risk</span></td>
                </tr>"""
            st.markdown(f"""
            <table class="stbl">
              <thead><tr>
                <th>Student ID</th><th>CGPA</th><th>AVG</th>
                <th>Attendance</th><th>Grade</th><th>Risk Level</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.success("No at-risk students in the current filter.")

    # ══════════════════════════════
    # LEARNING CURVE
    # ══════════════════════════════
    with t3:
        c1, c2 = st.columns([1,2])

        with c1:
            section("Pass vs Dropout Rate")
            fig = go.Figure(go.Pie(
                labels=["Pass","Dropout"],
                values=[pass_r, drop_r],
                hole=0.48,
                marker_colors=["#16a34a","#dc2626"],
                textfont_size=14,
                textinfo="label+percent"
            ))
            fig.update_layout(
                height=380,
                paper_bgcolor="#ffffff",
                font=dict(color="#1e293b"),
                legend=dict(font=dict(color="#475569"),
                            bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Student Outcome Rate",
                           font=dict(color="#1e293b", size=13)),
                margin=dict(t=50,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;
                 border-radius:8px;padding:.85rem 1rem;margin-top:.5rem;">
              <div style="display:flex;justify-content:space-between;
                   margin-bottom:.4rem;">
                <span style="color:#475569;font-size:.85rem;">Pass Rate</span>
                <strong style="color:#16a34a">{pass_r:.1f}%</strong>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <span style="color:#475569;font-size:.85rem;">Dropout Rate</span>
                <strong style="color:#dc2626">{drop_r:.1f}%</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            if QUIZZ:
                section("Quiz Score Trend per Cluster")
                rows = []
                for cls in ["High-Performing","Moderate","At-Risk"]:
                    grp = df_f[df_f["cluster"]==cls]
                    if not len(grp): continue
                    for q in QUIZZ:
                        rows.append({"Cluster":cls,"Quiz":q,
                                     "Mean Score":grp[q].mean()})
                if rows:
                    fig = px.line(pd.DataFrame(rows),
                                  x="Quiz", y="Mean Score",
                                  color="Cluster",
                                  color_discrete_map=CLR,
                                  markers=True,
                                  title="Average Quiz Score per Cluster Across Weeks")
                    fig.update_traces(line_width=2.5, marker_size=9)
                    fig.update_layout(
                        xaxis_title="Quiz / Week",
                        yaxis_title="Mean Score")
                    st.plotly_chart(dl(fig, 460), use_container_width=True)
            else:
                section("Performance Learning Curve")
                weeks = ["W1","W2","W3","W4","W5","W6"]
                base  = df_f["AVG"].mean()
                cur   = [base*f for f in [.55,.65,.72,.82,.90,1.0]]
                pred  = [base*f for f in [.50,.62,.75,.85,.93,1.05]]
                fig   = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weeks, y=cur, name="Current Progress",
                    mode="lines+markers",
                    line=dict(color="#2563eb", width=2.5),
                    marker=dict(size=9)))
                fig.add_trace(go.Scatter(
                    x=weeks, y=pred, name="Predicted Progress",
                    mode="lines+markers",
                    line=dict(color="#94a3b8", width=2, dash="dash"),
                    marker=dict(size=9)))
                fig.update_layout(
                    xaxis_title="Week",
                    yaxis_title="Score")
                st.plotly_chart(dl(fig, 460), use_container_width=True)

        section("AVG Score Distribution by Cluster")
        fig = px.histogram(df_f, x="AVG", color="cluster",
                           color_discrete_map=CLR, nbins=20,
                           barmode="overlay", opacity=0.75,
                           title="AVG Score Frequency Distribution")
        fig.update_layout(xaxis_title="AVG Score",
                          yaxis_title="Number of Students")
        st.plotly_chart(dl(fig, 400), use_container_width=True)

    # ══════════════════════════════
    # PERFORMANCE ANALYSIS
    # ══════════════════════════════
    with t4:
        grade_counts = df_f["Grade"].value_counts()
        total_g      = grade_counts.sum()

        c1, c2 = st.columns(2)
        with c1:
            section("Grade Distribution")
            fig = go.Figure(go.Pie(
                labels=grade_counts.index.tolist(),
                values=grade_counts.values.tolist(),
                hole=0.45,
                marker_colors=["#7c3aed","#16a34a","#2563eb","#dc2626"],
                textfont_size=14,
                textinfo="label+percent+value"
            ))
            fig.update_layout(
                height=420,
                paper_bgcolor="#ffffff",
                font=dict(color="#1e293b"),
                legend=dict(font=dict(color="#475569"),
                            bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Grade Distribution (O / A / B / F)",
                           font=dict(color="#1e293b", size=13)),
                margin=dict(t=50,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            section("Grades per Cluster")
            gcls = (df_f.groupby(["cluster","Grade"])
                        .size().reset_index(name="Count"))
            fig  = px.bar(gcls, x="cluster", y="Count",
                          color="Grade", barmode="group",
                          title="Grade Breakdown per Student Group",
                          color_discrete_map={
                              "O":"#7c3aed","A":"#16a34a",
                              "B":"#2563eb","F":"#dc2626"},
                          text="Count")
            fig.update_traces(textposition="outside",
                              marker_line_color="white",
                              marker_line_width=1)
            fig.update_layout(xaxis_title="Cluster",
                              yaxis_title="Number of Students")
            st.plotly_chart(dl(fig, 420), use_container_width=True)

        section("Feature-wise Box Plots per Cluster")
        key_feats = [f for f in
                     ["AVG","CGPA(%)","Attendance","QUIZ Attempt",
                      "Resource Visited","No. of views on forum"]
                     if f in df_f.columns]
        fig_b = make_subplots(rows=2, cols=3,
                              subplot_titles=key_feats,
                              vertical_spacing=0.18,
                              horizontal_spacing=0.08)
        for fi, feat in enumerate(key_feats):
            r_, c_ = fi//3+1, fi%3+1
            for cls, clr in CLR.items():
                grp = df_f[df_f["cluster"]==cls][feat].dropna()
                fig_b.add_trace(
                    go.Box(y=grp.values, name=cls,
                           marker_color=clr, line_color=clr,
                           showlegend=(fi==0), boxmean=True),
                    row=r_, col=c_)
        fig_b.update_layout(
            height=560,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#64748b", size=11),
            legend=dict(font=dict(color="#475569"),
                        bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=1.04))
        for ann in fig_b.layout.annotations:
            ann.font.color  = "#1e293b"
            ann.font.size   = 12
        for key in fig_b.layout:
            if key.startswith(("xaxis","yaxis")):
                fig_b.layout[key].update(
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#94a3b8", size=10))
        st.plotly_chart(fig_b, use_container_width=True)

    # ══════════════════════════════
    # CLUSTER ANALYSIS
    # ══════════════════════════════
    with t5:
        def scatter(df_in, ccol, title):
            hov = [c for c in ["Student ID","CGPA(%)","AVG",
                                "Attendance","Grade","dropout_risk"]
                   if c in df_in.columns]
            fig = px.scatter(df_in, x="px", y="py", color=ccol,
                             color_discrete_map=CLR, opacity=0.8,
                             hover_data=hov, title=title,
                             labels={"px":"Principal Component 1",
                                     "py":"Principal Component 2"})
            if "Student ID" in df_in.columns:
                for _, row in df_in.iterrows():
                    fig.add_annotation(
                        x=row["px"], y=row["py"],
                        text=str(row["Student ID"]),
                        showarrow=False,
                        font=dict(size=8, color="#475569"),
                        yshift=11)
            fig.update_traces(marker=dict(
                size=12, line=dict(width=1.5, color="white")))
            return dl(fig, 500)

        def mbadge(sv, dv, hp_n, ar_n):
            st.markdown(f"""
            <div class="mbadge">
              <div class="mb">
                <div class="mbv">{sv}</div>
                <div class="mbl">Silhouette Score (higher = better)</div>
              </div>
              <div class="mb">
                <div class="mbv">{dv}</div>
                <div class="mbl">Davies-Bouldin (lower = better)</div>
              </div>
              <div class="mb">
                <div class="mbv" style="color:#16a34a">{hp_n}</div>
                <div class="mbl">High-Performing Students</div>
              </div>
              <div class="mb">
                <div class="mbv" style="color:#dc2626">{ar_n}</div>
                <div class="mbl">At-Risk Students</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        ct1,ct2,ct3,ct4 = st.tabs(
            ["K-Means","GMM","Hierarchical","Algorithm Comparison"])

        with ct1:
            section("K-Means Clustering — PCA Scatter Plot")
            a,b = st.columns([3,2])
            with a:
                st.plotly_chart(
                    scatter(df_f,"km_cls","K-Means Student Clusters"),
                    use_container_width=True)
            with b:
                section("Cluster Size")
                sz = df_f["km_cls"].value_counts().reset_index()
                sz.columns = ["Cluster","Count"]
                fig = px.bar(sz, x="Cluster", y="Count",
                             color="Cluster",
                             color_discrete_map=CLR,
                             text="Count",
                             title="Number of Students per Cluster")
                fig.update_traces(textposition="outside",
                                  marker_line_color="white",
                                  marker_line_width=1.5)
                fig.update_layout(showlegend=False)
                st.plotly_chart(dl(fig, 500), use_container_width=True)
            mbadge(sil["K-Means"], dbs["K-Means"],
                   (df_f["km_cls"]=="High-Performing").sum(),
                   (df_f["km_cls"]=="At-Risk").sum())

        with ct2:
            section("GMM Clustering — PCA Scatter Plot")
            a,b = st.columns([3,2])
            with a:
                st.plotly_chart(
                    scatter(df_f,"gm_cls","GMM Student Clusters"),
                    use_container_width=True)
            with b:
                section("Assignment Confidence")
                fig = px.histogram(df_f, x="gm_prob", color="gm_cls",
                                   color_discrete_map=CLR, nbins=20,
                                   title="Cluster Assignment Confidence",
                                   barmode="overlay", opacity=0.75)
                fig.update_layout(
                    xaxis_title="Probability",
                    yaxis_title="Number of Students")
                st.plotly_chart(dl(fig, 500), use_container_width=True)
            mbadge(sil["GMM"], dbs["GMM"],
                   (df_f["gm_cls"]=="High-Performing").sum(),
                   (df_f["gm_cls"]=="At-Risk").sum())

        with ct3:
            section("Hierarchical Clustering")
            a,b = st.columns([3,2])
            with a:
                st.plotly_chart(
                    scatter(df_f,"hc_cls",
                            "Hierarchical Student Clusters"),
                    use_container_width=True)
            with b:
                section("Dendrogram — Ward Linkage")
                Z2     = linkage(X, method="ward")
                fd, ax = plt.subplots(figsize=(5,4.5))
                fd.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#f8fafc")
                lbls = (df["Student ID"].astype(str).tolist()
                        if "Student ID" in df.columns else None)
                dendrogram(Z2, ax=ax, labels=lbls,
                           color_threshold=Z2[-n_clusters+1,2],
                           above_threshold_color="#94a3b8",
                           leaf_font_size=6, leaf_rotation=90)
                ax.axhline(y=Z2[-n_clusters+1,2],
                           color="#dc2626", ls="--", lw=2,
                           label=f"Cut: {n_clusters} clusters")
                ax.set_title("Hierarchical Dendrogram",
                             color="#1e293b", fontsize=10,
                             fontweight="bold")
                ax.set_xlabel("Student ID", color="#64748b",
                              fontsize=9)
                ax.set_ylabel("Distance", color="#64748b",
                              fontsize=9)
                ax.tick_params(colors="#94a3b8")
                ax.legend(fontsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#e2e8f0")
                plt.tight_layout()
                buf = io.BytesIO()
                fd.savefig(buf, format="png", dpi=120,
                           bbox_inches="tight",
                           facecolor="#ffffff")
                buf.seek(0)
                b.image(buf, use_container_width=True)
                plt.close(fd)
            mbadge(sil["Hierarchical"], dbs["Hierarchical"],
                   (df_f["hc_cls"]=="High-Performing").sum(),
                   (df_f["hc_cls"]=="At-Risk").sum())

        with ct4:
            section("Algorithm Performance Comparison")
            best_s = max(sil, key=sil.get)
            best_d = min(dbs, key=dbs.get)
            rows   = ""
            for al in ["K-Means","GMM","Hierarchical"]:
                ck  = {"K-Means":"km_cls","GMM":"gm_cls",
                       "Hierarchical":"hc_cls"}[al]
                hp_ = (df[ck]=="High-Performing").sum()
                mo_ = (df[ck]=="Moderate").sum()
                ar_ = (df[ck]=="At-Risk").sum()
                sb  = '<span class="best-tag">Best</span>' \
                      if al==best_s else ""
                db_ = '<span class="best-tag">Best</span>' \
                      if al==best_d else ""
                rows += f"""<tr>
                  <td><strong>{al}</strong></td>
                  <td>{sil[al]}{sb}</td>
                  <td>{dbs[al]}{db_}</td>
                  <td><span style="color:#16a34a;font-weight:700">{hp_}</span></td>
                  <td><span style="color:#d97706;font-weight:700">{mo_}</span></td>
                  <td><span style="color:#dc2626;font-weight:700">{ar_}</span></td>
                </tr>"""
            st.markdown(f"""
            <table class="ctbl">
              <thead><tr>
                <th>Algorithm</th>
                <th>Silhouette Score</th>
                <th>Davies-Bouldin</th>
                <th style="color:#16a34a">High-Performing</th>
                <th style="color:#d97706">Moderate</th>
                <th style="color:#dc2626">At-Risk</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table><br>
            """, unsafe_allow_html=True)

            fc = make_subplots(1,2, subplot_titles=[
                "Silhouette Score (higher is better)",
                "Davies-Bouldin Score (lower is better)"])
            bc = ["#2563eb","#16a34a","#d97706"]
            fc.add_trace(go.Bar(
                x=list(sil.keys()), y=list(sil.values()),
                marker_color=bc, text=list(sil.values()),
                textposition="outside",
                marker_line_color="white",
                marker_line_width=1.5), row=1, col=1)
            fc.add_trace(go.Bar(
                x=list(dbs.keys()), y=list(dbs.values()),
                marker_color=bc, text=list(dbs.values()),
                textposition="outside",
                marker_line_color="white",
                marker_line_width=1.5), row=1, col=2)
            fc.update_layout(
                height=420, showlegend=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"))
            fc.update_annotations(font_color="#1e293b", font_size=13)
            for k in fc.layout:
                if k.startswith(("xaxis","yaxis")):
                    fc.layout[k].update(
                        gridcolor="#e2e8f0",
                        tickfont=dict(color="#64748b"))
            st.plotly_chart(fc, use_container_width=True)

            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                 border-radius:8px;padding:.85rem 1.2rem;margin-top:.5rem;">
              <span style="color:#475569;font-size:.85rem;">
              Recommended algorithm: </span>
              <strong style="color:#16a34a">{best_s}</strong>
              <span style="color:#94a3b8;font-size:.82rem;">
              &nbsp; (Silhouette = {sil[best_s]},
              Davies-Bouldin = {dbs[best_s]})</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════
    # COURSE OUTCOMES
    # ══════════════════════════════
    with t6:
        co = pd.DataFrame({
            "CO":["CO1: Academic Performance",
                  "CO2: Attendance",
                  "CO3: Prior Knowledge",
                  "CO4: Student Engagement",
                  "CO5: CGPA Attainment"],
            "Attainment":[
                df_f["AVG"].mean(),
                df_f["Attendance"].mean() * 10,
                df_f["Prior Knowledge of subject"].mean() * 20
                if "Prior Knowledge of subject" in df_f.columns else 0,
                ((df_f.get("Resource Visited",
                            pd.Series([0]*len(df_f))).mean() +
                  df_f.get("No. of Replies Posted",
                            pd.Series([0]*len(df_f))).mean()) / 2) * 20,
                df_f["CGPA(%)"].mean()
            ]
        })
        co["Attainment"] = co["Attainment"].clip(0,100).round(1)

        section("Course Outcome Attainment")
        co_cols = st.columns(5)
        for i, (_,row) in enumerate(co.iterrows()):
            with co_cols[i]:
                st.markdown(f"""
                <div class="co-card">
                  <div class="co-val">{row['Attainment']}%</div>
                  <div class="co-lbl">{row['CO']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Attainment Chart")
        fig_co = px.bar(co, x="CO", y="Attainment",
                        text="Attainment",
                        title="Course Outcome Attainment Levels (%)",
                        color="Attainment",
                        color_continuous_scale="RdYlGn",
                        range_color=[0,100])
        fig_co.update_traces(
            texttemplate="%{text}%", textposition="outside",
            marker_line_color="white", marker_line_width=1.5)
        fig_co.update_layout(
            yaxis=dict(range=[0,115]),
            xaxis_title="Course Outcome",
            yaxis_title="Attainment (%)")
        st.plotly_chart(dl(fig_co, 440), use_container_width=True)

        bco = co.loc[co["Attainment"].idxmax()]
        wco = co.loc[co["Attainment"].idxmin()]
        c1, c2 = st.columns(2)
        c1.success(
            f"Highest Attainment: {bco['CO']} — {bco['Attainment']}%")
        c2.warning(
            f"Needs Improvement: {wco['CO']} — {wco['Attainment']}%")

    # ── EXPORT ──
    st.divider()
    ec = [c for c in
          (["Student ID"] if "Student ID" in df.columns else []) +
          FEAT + ["Grade","cluster","km_cls","gm_cls",
                  "hc_cls","dropout_risk"]
          if c in df.columns]
    st.download_button(
        "Download Results as CSV",
        df[ec].to_csv(index=False),
        "student_cluster_results.csv",
        "text/csv")

else:
    st.markdown("""
    <div style="background:#ffffff;border:2px dashed #e2e8f0;
                border-radius:14px;padding:4rem;text-align:center;
                margin-top:3rem;box-shadow:0 1px 4px rgba(0,0,0,.06);">
      <h2 style="color:#1e293b;margin:.5rem 0;">
        Upload Student Data to Begin</h2>
      <p style="color:#64748b;max-width:480px;margin:.5rem auto 0;
                line-height:1.7;">
        Use the sidebar to upload your student CSV file.
        The dashboard supports both
        <strong>Mtech_Selected_ML_Ready.csv</strong> and
        <strong>Mtechlabeldataset.csv</strong>.
      </p>
      <div style="margin-top:2rem;background:#f8fafc;border:1px solid #e2e8f0;
                  border-radius:8px;padding:1rem 1.5rem;display:inline-block;
                  text-align:left;">
        <p style="color:#64748b;font-size:.83rem;margin:0;line-height:1.8;">
          <strong style="color:#475569">Required columns:</strong><br>
          Roll Number / Student ID &nbsp;·&nbsp; CGPA(%) &nbsp;·&nbsp;
          AVG &nbsp;·&nbsp; Attendance &nbsp;·&nbsp; QUIZ 1-6 &nbsp;·&nbsp;
          10th Score(%) &nbsp;·&nbsp; 12th Score(%)
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)
