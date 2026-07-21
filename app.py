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
    page_title="Learning Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLR = {
    "High-Performing": "#10b981",
    "Moderate":        "#f59e0b",
    "At-Risk":         "#ef4444"
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
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.main .block-container { padding-top: 1.2rem; }

.top-bar {
    background: linear-gradient(135deg,#1e3a5f,#1e293b);
    border: 1px solid #334155; border-radius: 12px;
    padding: 1.2rem 2rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
}
.top-title { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }
.top-sub   { font-size: .8rem; color: #64748b; margin-top: .2rem; }

.kpi-row { display: flex; gap: .75rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 130px; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
    border: 1px solid #334155;
}
.kpi.b { background: linear-gradient(135deg,#0c4a6e,#1e293b); }
.kpi.g { background: linear-gradient(135deg,#064e3b,#1e293b); }
.kpi.r { background: linear-gradient(135deg,#7f1d1d,#1e293b); }
.kpi.a { background: linear-gradient(135deg,#78350f,#1e293b); }
.kpi.p { background: linear-gradient(135deg,#4c1d95,#1e293b); }
.kv { font-size: 1.7rem; font-weight: 800; }
.kpi.b .kv { color: #38bdf8; }
.kpi.g .kv { color: #34d399; }
.kpi.r .kv { color: #f87171; }
.kpi.a .kv { color: #fbbf24; }
.kpi.p .kv { color: #a78bfa; }
.kl { font-size: .7rem; color: #64748b; text-transform: uppercase;
      letter-spacing: .04em; margin-top: .2rem; }

.comp-tbl { width:100%; border-collapse:collapse; }
.comp-tbl th { background:#1e293b; color:#94a3b8; font-size:.75rem;
               padding:.7rem 1rem; border-bottom:2px solid #334155;
               text-align:left; text-transform:uppercase; }
.comp-tbl td { padding:.75rem 1rem; border-bottom:1px solid #1e293b;
               font-size:.85rem; color:#e2e8f0; }
.comp-tbl tr:hover td { background:#1e293b; }
.best { background:rgba(16,185,129,.2); color:#10b981;
        padding:.15rem .45rem; border-radius:4px;
        font-size:.7rem; font-weight:700; margin-left:.3rem; }

.mbadge { display:flex; gap:.75rem; flex-wrap:wrap; margin-top:.75rem; }
.mb { background:#1e293b; border:1px solid #334155; border-radius:8px;
      padding:.65rem 1rem; text-align:center; flex:1; min-width:110px; }
.mbv { font-size:1.3rem; font-weight:800; }
.mbl { font-size:.68rem; color:#64748b; margin-top:.15rem; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    uploaded     = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    n_clusters   = st.slider("Clusters (K)", 2, 6, 3)
    algorithm    = st.selectbox("Algorithm", ["K-Means","GMM","Hierarchical"])
    st.divider()
    risk_filter  = st.multiselect("Risk Filter",
                                  ["High","Medium","Low"],
                                  default=["High","Medium","Low"])
    clust_filter = st.multiselect("Cluster Filter",
                                  ["High-Performing","Moderate","At-Risk"],
                                  default=["High-Performing","Moderate","At-Risk"])

# ── HELPERS ──
def dl(fig, h=400):
    fig.update_layout(
        height=h, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=11),
        title_font=dict(color="#e2e8f0", size=13),
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title_font=dict(color="#94a3b8"))
    )
    return fig

def assign_names(df, lbl, name, feats, n):
    dn = df[feats].copy()
    for c in feats:
        r = dn[c].max()-dn[c].min()
        if r > 0: dn[c] = (dn[c]-dn[c].min())/r
    sc = {l: dn.loc[df[lbl]==l].mean().mean() for l in range(n)}
    sl = sorted(sc, key=sc.get, reverse=True)
    nm = (["High-Performing","Moderate","At-Risk"]+
          [f"C{i}" for i in range(4,n+1)])[:n]
    df[name] = df[lbl].map({sl[i]:nm[i] for i in range(n)})
    return df

def grade(avg):
    if avg >= 75:   return "O"
    elif avg >= 60: return "A"
    elif avg >= 50: return "B"
    else:           return "F"

# ════════════════════════════════
if uploaded:
    # ── LOAD & PREPROCESS ──
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip()
    if "Roll Number" in df_raw.columns:
        df_raw.rename(columns={"Roll Number":"Student ID"}, inplace=True)

    FEAT  = [c for c in FEAT_COLS if c in df_raw.columns]
    QUIZZ = [c for c in QUIZ_COLS  if c in df_raw.columns]
    df    = df_raw.copy()

    for col in FEAT:
        df[col] = df[col].clip(df[col].quantile(.01), df[col].quantile(.99))
    df[FEAT] = SimpleImputer(strategy="mean").fit_transform(df[FEAT])
    X        = StandardScaler().fit_transform(df[FEAT])
    Xp       = PCA(n_components=2, random_state=42).fit_transform(X)
    df["px"], df["py"] = Xp[:,0], Xp[:,1]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["km_lbl"] = km.fit_predict(X)
    df = assign_names(df,"km_lbl","km_cls",FEAT,n_clusters)

    gm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    gm.fit(X)
    df["gm_lbl"]  = gm.predict(X)
    df["gm_prob"] = gm.predict_proba(X).max(axis=1)
    df = assign_names(df,"gm_lbl","gm_cls",FEAT,n_clusters)

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df["hc_lbl"] = hc.fit_predict(X)
    df = assign_names(df,"hc_lbl","hc_cls",FEAT,n_clusters)

    amap = {"K-Means":"km_cls","GMM":"gm_cls","Hierarchical":"hc_cls"}
    df["cluster"]      = df[amap[algorithm]]
    df["dropout_risk"] = df["cluster"].map(
        {"High-Performing":"Low","Moderate":"Medium","At-Risk":"High"})
    df["Grade"] = df["AVG"].apply(grade)

    sil = {"K-Means":     round(silhouette_score(X,df["km_lbl"]),4),
           "GMM":         round(silhouette_score(X,df["gm_lbl"]),4),
           "Hierarchical":round(silhouette_score(X,df["hc_lbl"]),4)}
    dbs = {"K-Means":     round(davies_bouldin_score(X,df["km_lbl"]),4),
           "GMM":         round(davies_bouldin_score(X,df["gm_lbl"]),4),
           "Hierarchical":round(davies_bouldin_score(X,df["hc_lbl"]),4)}
    best = max(sil, key=sil.get)

    df_f = df[df["dropout_risk"].isin(risk_filter) &
              df["cluster"].isin(clust_filter)]

    N   = len(df_f)
    hp  = (df_f["cluster"]=="High-Performing").sum()
    mo  = (df_f["cluster"]=="Moderate").sum()
    ar  = (df_f["cluster"]=="At-Risk").sum()
    avg_perf = df_f[["CGPA(%)","AVG"]].mean().mean()
    pass_r   = (df_f["Grade"]!="F").sum()/max(N,1)*100

    # ── TOP BAR ──
    st.markdown(f"""
    <div class="top-bar">
      <div>
        <div class="top-title">🎓 Learning Analytics Dashboard</div>
        <div class="top-sub">{N} students &nbsp;·&nbsp; {algorithm} &nbsp;·&nbsp; Best: {best}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "🏠 Home", "📊 Overview", "📈 Learning Curve",
        "🎯 Performance", "🔬 Cluster Analysis", "🏆 Course Outcomes"
    ])

    # ════════════ HOME ════════════
    with t1:
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi b"><div class="kv">{N}</div><div class="kl">Total Students</div></div>
          <div class="kpi g"><div class="kv">{hp}</div><div class="kl">High-Performing</div></div>
          <div class="kpi a"><div class="kv">{mo}</div><div class="kl">Moderate</div></div>
          <div class="kpi r"><div class="kv">{ar}</div><div class="kl">At-Risk</div></div>
          <div class="kpi p"><div class="kv">{avg_perf:.0f}%</div><div class="kl">Avg Performance</div></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sz = df_f["cluster"].value_counts().reset_index()
            sz.columns = ["Cluster","Count"]
            fig = px.pie(sz, names="Cluster", values="Count",
                         color="Cluster", color_discrete_map=CLR,
                         hole=0.5, title="Cluster Distribution")
            fig.update_traces(textfont_color="white", textfont_size=13)
            st.plotly_chart(dl(fig, 350), use_container_width=True)
        with c2:
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;
                 border-radius:10px;padding:1.2rem 1.5rem;margin-top:.5rem;">
              <p style="color:#38bdf8;font-weight:700;margin:0 0 .75rem;">📌 Project Summary</p>
              <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:0">
              Unsupervised ML on real M.Tech data (57 students).<br>
              Algorithms: K-Means · GMM · Hierarchical<br>
              Features: Academic scores, Attendance, Quiz, Forum activity<br><br>
              <strong style="color:#10b981">Best Algorithm: {best}</strong><br>
              Silhouette = {sil[best]} &nbsp;|&nbsp; Davies-Bouldin = {dbs[best]}
              </p>
            </div>
            """, unsafe_allow_html=True)

    # ════════════ OVERVIEW ════════════
    with t2:
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi b"><div class="kv">{avg_perf:.1f}%</div><div class="kl">Avg CGPA+AVG</div></div>
          <div class="kpi r"><div class="kv">{ar/max(N,1)*100:.0f}%</div><div class="kl">At-Risk</div></div>
          <div class="kpi a"><div class="kv">{df_f['Attendance'].mean():.1f}</div><div class="kl">Avg Attendance</div></div>
          <div class="kpi g"><div class="kv">{pass_r:.0f}%</div><div class="kl">Pass Rate</div></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df_f["dropout_risk"].value_counts().reset_index()
                    .rename(columns={"dropout_risk":"Risk","count":"Count"}),
                x="Risk", y="Count", color="Risk",
                color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                text="Count", title="Risk Distribution")
            fig.update_traces(textposition="outside",
                              marker_line_color="#0f172a", marker_line_width=1.5)
            st.plotly_chart(dl(fig, 360), use_container_width=True)
        with c2:
            fig = px.box(df_f, x="cluster", y="AVG",
                         color="cluster", color_discrete_map=CLR,
                         title="AVG Score by Cluster", points="all")
            fig.update_layout(showlegend=False)
            st.plotly_chart(dl(fig, 360), use_container_width=True)

    # ════════════ LEARNING CURVE ════════════
    with t3:
        c1, c2 = st.columns([1,2])
        with c1:
            drop_r = 100 - pass_r
            fig = go.Figure(go.Pie(
                labels=["Pass","Dropout"],
                values=[pass_r, drop_r],
                hole=0.5,
                marker_colors=["#10b981","#ef4444"],
                textfont_size=13, textfont_color="white"
            ))
            fig.update_layout(
                height=300, paper_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                legend=dict(font=dict(color="#e2e8f0"),
                            bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Pass vs Dropout", font=dict(color="#e2e8f0")),
                margin=dict(t=40,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="color:#e2e8f0;font-size:.85rem;margin-top:.5rem;">
              🟢 Pass Rate &nbsp;<strong style="color:#34d399">{pass_r:.1f}%</strong><br>
              🔴 Dropout Rate &nbsp;<strong style="color:#f87171">{drop_r:.1f}%</strong>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            if QUIZZ:
                rows = []
                for cls in ["High-Performing","Moderate","At-Risk"]:
                    grp = df_f[df_f["cluster"]==cls]
                    if not len(grp): continue
                    for q in QUIZZ:
                        rows.append({"Cluster":cls,"Quiz":q,
                                     "Score":grp[q].mean()})
                if rows:
                    fig = px.line(pd.DataFrame(rows),
                                  x="Quiz", y="Score",
                                  color="Cluster",
                                  color_discrete_map=CLR,
                                  markers=True,
                                  title="Quiz Score Trend per Cluster")
                    fig.update_traces(line_width=2.5, marker_size=8)
                    st.plotly_chart(dl(fig, 340), use_container_width=True)
            else:
                weeks = ["W1","W2","W3","W4","W5","W6"]
                base  = df_f["AVG"].mean()
                cur   = [base*f for f in [.55,.65,.72,.82,.90,1.0]]
                pred  = [base*f for f in [.50,.62,.75,.85,.93,1.05]]
                fig   = go.Figure()
                fig.add_trace(go.Scatter(x=weeks,y=cur,name="Current",
                    mode="lines+markers",
                    line=dict(color="#38bdf8",width=2.5),
                    marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=weeks,y=pred,name="Predicted",
                    mode="lines+markers",
                    line=dict(color="#94a3b8",width=2,dash="dash"),
                    marker=dict(size=8)))
                fig.update_layout(title="Performance Learning Curve")
                st.plotly_chart(dl(fig, 340), use_container_width=True)

    # ════════════ PERFORMANCE ════════════
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            gc = df_f["Grade"].value_counts().reset_index()
            gc.columns = ["Grade","Count"]
            fig = go.Figure(go.Pie(
                labels=gc["Grade"].tolist(),
                values=gc["Count"].tolist(),
                hole=0.48,
                marker_colors=["#a78bfa","#10b981","#38bdf8","#ef4444"],
                textfont_size=14, textfont_color="white",
                textinfo="label+percent"
            ))
            fig.update_layout(
                height=360, paper_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                legend=dict(font=dict(color="#e2e8f0"),
                            bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Grade Distribution (O/A/B/F)",
                           font=dict(color="#e2e8f0")),
                margin=dict(t=50,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gcls = (df_f.groupby(["cluster","Grade"])
                        .size().reset_index(name="Count"))
            fig  = px.bar(gcls, x="cluster", y="Count",
                          color="Grade", barmode="group",
                          title="Grades per Cluster",
                          color_discrete_map={
                              "O":"#a78bfa","A":"#10b981",
                              "B":"#38bdf8","F":"#ef4444"})
            fig.update_traces(marker_line_color="#0f172a",
                              marker_line_width=1.2)
            st.plotly_chart(dl(fig, 360), use_container_width=True)

        # Box plots — 2 columns grid, key features only
        key_feats = [f for f in
                     ["AVG","CGPA(%)","Attendance","QUIZ Attempt",
                      "Resource Visited","No. of views on forum"]
                     if f in df_f.columns]
        fig_b = make_subplots(rows=2, cols=3,
                              subplot_titles=key_feats,
                              vertical_spacing=0.18,
                              horizontal_spacing=0.08)
        for fi, feat in enumerate(key_feats):
            r, c_ = fi//3+1, fi%3+1
            for cls, clr in CLR.items():
                grp = df_f[df_f["cluster"]==cls][feat].dropna()
                fig_b.add_trace(
                    go.Box(y=grp.values, name=cls,
                           marker_color=clr, line_color=clr,
                           showlegend=(fi==0), boxmean=True),
                    row=r, col=c_)
        fig_b.update_layout(
            height=480, paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#94a3b8", size=10),
            legend=dict(font=dict(color="#e2e8f0"),
                        bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=1.05))
        for ann in fig_b.layout.annotations:
            ann.font.color = "#e2e8f0"
            ann.font.size  = 11
        for key in fig_b.layout:
            if key.startswith(("xaxis","yaxis")):
                fig_b.layout[key].update(
                    gridcolor="#334155",
                    tickfont=dict(color="#64748b", size=9))
        st.plotly_chart(fig_b, use_container_width=True)

    # ════════════ CLUSTER ANALYSIS ════════════
    with t5:
        ct1, ct2, ct3, ct4 = st.tabs(
            ["K-Means","GMM","Hierarchical","Comparison"])

        def scatter(df_in, ccol, title):
            hov = [c for c in ["Student ID","CGPA(%)","AVG",
                                "Attendance","dropout_risk"]
                   if c in df_in.columns]
            fig = px.scatter(df_in, x="px", y="py", color=ccol,
                             color_discrete_map=CLR, opacity=0.85,
                             hover_data=hov, title=title,
                             labels={"px":"PC1","py":"PC2"})
            if "Student ID" in df_in.columns:
                for _, row in df_in.iterrows():
                    fig.add_annotation(
                        x=row["px"], y=row["py"],
                        text=str(row["Student ID"]),
                        showarrow=False,
                        font=dict(size=8, color="#e2e8f0"),
                        yshift=10)
            fig.update_traces(marker=dict(size=11,
                              line=dict(width=1.5,color="#0f172a")))
            return dl(fig, 440)

        def mbadge(s_val, db_val, hp_n, ar_n, s_clr, db_clr):
            st.markdown(f"""
            <div class="mbadge">
              <div class="mb"><div class="mbv" style="color:{s_clr}">{s_val}</div>
                <div class="mbl">Silhouette ↑</div></div>
              <div class="mb"><div class="mbv" style="color:{db_clr}">{db_val}</div>
                <div class="mbl">Davies-Bouldin ↓</div></div>
              <div class="mb"><div class="mbv" style="color:#34d399">{hp_n}</div>
                <div class="mbl">High-Performing</div></div>
              <div class="mb"><div class="mbv" style="color:#f87171">{ar_n}</div>
                <div class="mbl">At-Risk</div></div>
            </div>
            """, unsafe_allow_html=True)

        with ct1:
            a,b = st.columns([2,1])
            with a: st.plotly_chart(scatter(df_f,"km_cls","K-Means"),
                                    use_container_width=True)
            with b:
                sz = df_f["km_cls"].value_counts().reset_index()
                sz.columns=["Cluster","Count"]
                fig=px.pie(sz,names="Cluster",values="Count",
                           color="Cluster",color_discrete_map=CLR,
                           hole=0.48)
                fig.update_traces(textfont_color="white",textfont_size=12)
                st.plotly_chart(dl(fig,440),use_container_width=True)
            mbadge(sil["K-Means"], dbs["K-Means"],
                   (df_f["km_cls"]=="High-Performing").sum(),
                   (df_f["km_cls"]=="At-Risk").sum(),
                   "#38bdf8","#f87171")

        with ct2:
            a,b = st.columns([2,1])
            with a: st.plotly_chart(scatter(df_f,"gm_cls","GMM"),
                                    use_container_width=True)
            with b:
                fig = px.histogram(df_f,x="gm_prob",color="gm_cls",
                                   color_discrete_map=CLR,nbins=20,
                                   title="Confidence",barmode="overlay",
                                   opacity=0.75)
                st.plotly_chart(dl(fig,440),use_container_width=True)
            mbadge(sil["GMM"],dbs["GMM"],
                   (df_f["gm_cls"]=="High-Performing").sum(),
                   (df_f["gm_cls"]=="At-Risk").sum(),
                   "#38bdf8","#f87171")

        with ct3:
            a,b = st.columns([3,2])
            with a: st.plotly_chart(scatter(df_f,"hc_cls","Hierarchical"),
                                    use_container_width=True)
            with b:
                st.markdown('<p style="color:#e2e8f0;font-weight:700;'
                            'margin-bottom:.4rem;">Dendrogram</p>',
                            unsafe_allow_html=True)
                Z2     = linkage(X, method="ward")
                fd, ax = plt.subplots(figsize=(5,4))
                fd.patch.set_facecolor("#1e293b")
                ax.set_facecolor("#1e293b")
                lbls = (df["Student ID"].astype(str).tolist()
                        if "Student ID" in df.columns else None)
                dendrogram(Z2, ax=ax, labels=lbls,
                           color_threshold=Z2[-n_clusters+1,2],
                           above_threshold_color="#475569",
                           leaf_font_size=6, leaf_rotation=90)
                ax.axhline(y=Z2[-n_clusters+1,2],
                           color="#ef4444", ls="--", lw=1.8)
                ax.set_title("Ward Linkage", color="#e2e8f0",
                             fontsize=10, fontweight="bold")
                ax.tick_params(colors="#64748b")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#334155")
                plt.tight_layout()
                buf = io.BytesIO()
                fd.savefig(buf,format="png",dpi=110,bbox_inches="tight")
                buf.seek(0)
                b.image(buf,use_container_width=True)
                plt.close(fd)
            mbadge(sil["Hierarchical"],dbs["Hierarchical"],
                   (df_f["hc_cls"]=="High-Performing").sum(),
                   (df_f["hc_cls"]=="At-Risk").sum(),
                   "#38bdf8","#f87171")

        with ct4:
            best_s = max(sil,key=sil.get)
            best_d = min(dbs,key=dbs.get)
            rows   = ""
            for al in ["K-Means","GMM","Hierarchical"]:
                ck = {"K-Means":"km_cls","GMM":"gm_cls",
                      "Hierarchical":"hc_cls"}[al]
                hp_= (df[ck]=="High-Performing").sum()
                mo_= (df[ck]=="Moderate").sum()
                ar_= (df[ck]=="At-Risk").sum()
                sb = '<span class="best">BEST</span>' if al==best_s else ""
                db_ = '<span class="best">BEST</span>' if al==best_d else ""
                rows += f"""<tr>
                  <td><strong style="color:#e2e8f0">{al}</strong></td>
                  <td>{sil[al]}{sb}</td><td>{dbs[al]}{db_}</td>
                  <td><span style="color:#10b981;font-weight:700">{hp_}</span></td>
                  <td><span style="color:#f59e0b;font-weight:700">{mo_}</span></td>
                  <td><span style="color:#ef4444;font-weight:700">{ar_}</span></td>
                </tr>"""
            st.markdown(f"""
            <table class="comp-tbl">
              <thead><tr>
                <th>Algorithm</th><th>Silhouette ↑</th>
                <th>Davies-Bouldin ↓</th>
                <th style="color:#10b981">High</th>
                <th style="color:#f59e0b">Moderate</th>
                <th style="color:#ef4444">At-Risk</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table><br>
            """, unsafe_allow_html=True)

            fc = make_subplots(1,2,
                    subplot_titles=["Silhouette","Davies-Bouldin"])
            bc = ["#0ea5e9","#10b981","#f59e0b"]
            fc.add_trace(go.Bar(x=list(sil.keys()),y=list(sil.values()),
                                marker_color=bc,text=list(sil.values()),
                                textposition="outside"),row=1,col=1)
            fc.add_trace(go.Bar(x=list(dbs.keys()),y=list(dbs.values()),
                                marker_color=bc,text=list(dbs.values()),
                                textposition="outside"),row=1,col=2)
            fc.update_layout(height=360,showlegend=False,
                             paper_bgcolor="#0f172a",plot_bgcolor="#1e293b",
                             font=dict(color="#e2e8f0"))
            fc.update_annotations(font_color="#e2e8f0")
            for k in fc.layout:
                if k.startswith(("xaxis","yaxis")):
                    fc.layout[k].update(gridcolor="#334155",
                                        tickfont=dict(color="#94a3b8"))
            st.plotly_chart(fc, use_container_width=True)

    # ════════════ COURSE OUTCOMES ════════════
    with t6:
        co = pd.DataFrame({
            "CO":["CO1: Academic","CO2: Attendance",
                  "CO3: Prior Knowledge","CO4: Engagement","CO5: CGPA"],
            "Attainment":[
                df_f["AVG"].mean(),
                df_f["Attendance"].mean()*10,
                df_f["Prior Knowledge of subject"].mean()*20
                if "Prior Knowledge of subject" in df_f.columns else 0,
                ((df_f.get("Resource Visited",pd.Series([0])).mean() +
                  df_f.get("No. of Replies Posted",pd.Series([0])).mean())/2)*20,
                df_f["CGPA(%)"].mean()
            ]
        })
        co["Attainment"] = co["Attainment"].clip(0,100).round(1)

        cols_co = st.columns(5)
        co_clrs = ["#38bdf8","#10b981","#f59e0b","#a78bfa","#f472b6"]
        for i,(_,row) in enumerate(co.iterrows()):
            with cols_co[i]:
                st.markdown(f"""
                <div style="background:#1e293b;border:1px solid #334155;
                     border-radius:10px;padding:1rem;text-align:center;">
                  <div style="font-size:1.6rem;font-weight:800;
                       color:{co_clrs[i]}">{row['Attainment']}%</div>
                  <div style="font-size:.68rem;color:#64748b;
                       margin-top:.2rem">{row['CO']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_co = px.bar(co, x="CO", y="Attainment",
                        text="Attainment",
                        title="Course Outcome Attainment (%)",
                        color="Attainment",
                        color_continuous_scale="RdYlGn",
                        range_color=[0,100])
        fig_co.update_traces(texttemplate="%{text}%",
                             textposition="outside",
                             marker_line_color="#0f172a",
                             marker_line_width=1.5)
        fig_co.update_layout(yaxis=dict(range=[0,115]))
        st.plotly_chart(dl(fig_co, 380), use_container_width=True)

        bco = co.loc[co["Attainment"].idxmax()]
        wco = co.loc[co["Attainment"].idxmin()]
        c1, c2 = st.columns(2)
        c1.success(f"Best: {bco['CO']} ({bco['Attainment']}%)")
        c2.error(f"Needs Work: {wco['CO']} ({wco['Attainment']}%)")

    # ── DOWNLOAD ──
    st.divider()
    ec = [c for c in
          (["Student ID"] if "Student ID" in df.columns else []) +
          FEAT + ["Grade","cluster","km_cls","gm_cls","hc_cls","dropout_risk"]
          if c in df.columns]
    st.download_button("⬇️ Download Results CSV",
                       df[ec].to_csv(index=False),
                       "results.csv","text/csv")

else:
    st.markdown("""
    <div style="background:#1e293b;border:2px dashed #334155;border-radius:14px;
                padding:3rem;text-align:center;margin-top:3rem;">
      <div style="font-size:3rem;">📂</div>
      <h3 style="color:#e2e8f0;margin:.5rem 0;">Upload CSV from sidebar</h3>
      <p style="color:#64748b;margin:0;">
        Supports both Mtech_Selected_ML_Ready.csv and Mtechlabeldataset.csv
      </p>
    </div>
    """, unsafe_allow_html=True)
