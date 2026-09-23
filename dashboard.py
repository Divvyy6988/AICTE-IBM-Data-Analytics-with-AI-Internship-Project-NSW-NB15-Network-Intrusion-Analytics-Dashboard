"""
UNSW-NB15 Network Intrusion Analytics Dashboard
================================================
Single-file Streamlit application — analysis only, no prediction model.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UNSW-NB15 Network Intrusion Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-sub   { font-size: 0.8rem;  color: #64748b; margin-top: 2px; }

    /* Section headers */
    .section-header {
        font-size: 1.25rem; font-weight: 600; color: #f1f5f9;
        border-left: 4px solid #38bdf8; padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* Risk / opportunity badges */
    .badge-red    { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:4px 10px; font-size:0.8rem; }
    .badge-yellow { background:#713f12; color:#fde68a; border-radius:6px; padding:4px 10px; font-size:0.8rem; }
    .badge-green  { background:#14532d; color:#86efac; border-radius:6px; padding:4px 10px; font-size:0.8rem; }

    /* Divider */
    hr { border-color: #334155 !important; }

    /* Metric delta override */
    [data-testid="metric-container"] { background:#1e293b; border-radius:10px; padding:12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Data loading & cleaning
# ─────────────────────────────────────────────
DATASET_FILE = "UNSW_NB15_training-set.csv"

NUMERIC_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
]

ATTACK_PALETTE = {
    "Normal":        "#22d3ee",
    "Generic":       "#f97316",
    "Exploits":      "#ef4444",
    "Fuzzers":       "#a855f7",
    "DoS":           "#ec4899",
    "Reconnaissance":"#eab308",
    "Analysis":      "#3b82f6",
    "Backdoor":      "#84cc16",
    "Shellcode":     "#14b8a6",
    "Worms":         "#f43f5e",
}


@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # ── 1. Strip whitespace from string columns ──
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── 2. Standardise attack_cat ──
    df["attack_cat"] = df["attack_cat"].replace(
        {"": "Normal", "nan": "Normal", " Backdoors": "Backdoor"}
    )

    # ── 3. Coerce numeric columns ──
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Fill numeric NaNs with column median ──
    for col in NUMERIC_COLS:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # ── 5. Remove exact duplicate rows ──
    df = df.drop_duplicates()

    # ── 6. Label as integer ──
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # ── 7. Derived columns ──
    df["is_attack"] = df["label"].map({0: "Normal", 1: "Attack"})
    df["total_bytes"] = df["sbytes"] + df["dbytes"]
    df["total_pkts"]  = df["spkts"]  + df["dpkts"]

    return df


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def plotly_dark_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=12),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#1e293b", linecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", linecolor="#334155")
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ UNSW-NB15 Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🔍 Attack Type Analysis",
            "⚡ Risk, Opportunity & Action",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Dataset: UNSW-NB15 Training Set<br>"
        "82,332 network flow records<br>"
        "45 features | 10 attack categories</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#475569'>Source: "
        "<a href='https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15' "
        "style='color:#38bdf8'>Kaggle Dataset</a></small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    df = load_data(DATASET_FILE)
except FileNotFoundError:
    st.error(
        f"❌ Dataset file **{DATASET_FILE}** not found. "
        "Place `UNSW_NB15_training-set.csv` in the same directory as this script."
    )
    st.stop()

# Pre-compute shared aggregates
attack_df  = df[df["label"] == 1].copy()
normal_df  = df[df["label"] == 0].copy()
cat_counts = df[df["label"] == 1]["attack_cat"].value_counts().reset_index()
cat_counts.columns = ["attack_cat", "count"]

total_records = len(df)
n_attacks     = int(df["label"].sum())
n_normal      = total_records - n_attacks
pct_attack    = n_attacks / total_records * 100
pct_normal    = 100 - pct_attack
top3_cats     = cat_counts.head(3)["attack_cat"].tolist()
avg_attack_dur = attack_df["dur"].mean()

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown(
        "High-level KPIs and summary statistics derived from the **UNSW-NB15** "
        "network intrusion training dataset."
    )
    st.markdown("---")

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value">{fmt_num(total_records)}</p>'
            f'<p class="kpi-label">Total Records</p>'
            f'<p class="kpi-sub">After deduplication</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#f97316">{pct_attack:.1f}%</p>'
            f'<p class="kpi-label">Attack Traffic</p>'
            f'<p class="kpi-sub">{fmt_num(n_attacks)} malicious flows</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#22d3ee">{pct_normal:.1f}%</p>'
            f'<p class="kpi-label">Normal Traffic</p>'
            f'<p class="kpi-sub">{fmt_num(n_normal)} benign flows</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#a855f7">{avg_attack_dur:.4f}s</p>'
            f'<p class="kpi-label">Avg Attack Duration</p>'
            f'<p class="kpi-sub">Mean flow duration (attacks)</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Top 3 attack categories callout ──
    st.markdown('<p class="section-header">🏆 Top 3 Attack Categories</p>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    icons = ["🥇", "🥈", "🥉"]
    colors = ["#f97316", "#94a3b8", "#b45309"]
    for col, (_, row), icon, color in zip(
        [t1, t2, t3], top3_cats[:3] if False else cat_counts.head(3).iterrows(), icons, colors
    ):
        with col:
            pct = row["count"] / n_attacks * 100
            st.markdown(
                f'<div class="kpi-card">'
                f'<p style="font-size:2rem;margin:0">{icon}</p>'
                f'<p class="kpi-value" style="color:{color};font-size:1.5rem">{row["attack_cat"]}</p>'
                f'<p class="kpi-label">{fmt_num(row["count"])} flows ({pct:.1f}% of attacks)</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Charts row 1 ──
    st.markdown('<p class="section-header">Traffic Composition & Attack Distribution</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        # Donut: normal vs attack
        pie_df = pd.DataFrame({
            "Type": ["Normal", "Attack"],
            "Count": [n_normal, n_attacks],
        })
        fig_donut = px.pie(
            pie_df,
            names="Type",
            values="Count",
            hole=0.55,
            color="Type",
            color_discrete_map={"Normal": "#22d3ee", "Attack": "#f97316"},
            title="Normal vs Attack Traffic",
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        plotly_dark_layout(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        # Horizontal bar: attack categories
        bar_df = cat_counts.sort_values("count")
        colors_bar = [ATTACK_PALETTE.get(c, "#64748b") for c in bar_df["attack_cat"]]
        fig_bar = go.Figure(
            go.Bar(
                x=bar_df["count"],
                y=bar_df["attack_cat"],
                orientation="h",
                marker_color=colors_bar,
                text=bar_df["count"].apply(fmt_num),
                textposition="outside",
            )
        )
        fig_bar.update_layout(title="Attack Category Frequency", xaxis_title="Flow Count", yaxis_title="")
        plotly_dark_layout(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Charts row 2 ──
    col_c, col_d = st.columns(2)

    with col_c:
        # Protocol breakdown
        proto_df = (
            df.groupby(["proto", "is_attack"])
            .size()
            .reset_index(name="count")
        )
        top_protos = df["proto"].value_counts().head(6).index.tolist()
        proto_df = proto_df[proto_df["proto"].isin(top_protos)]
        fig_proto = px.bar(
            proto_df,
            x="proto",
            y="count",
            color="is_attack",
            barmode="group",
            color_discrete_map={"Normal": "#22d3ee", "Attack": "#f97316"},
            title="Top Protocols: Normal vs Attack",
            labels={"proto": "Protocol", "count": "Flow Count", "is_attack": ""},
        )
        plotly_dark_layout(fig_proto)
        st.plotly_chart(fig_proto, use_container_width=True)

    with col_d:
        # Duration box plot
        dur_sample = df[df["dur"] < df["dur"].quantile(0.99)].sample(
            min(5000, len(df)), random_state=42
        )
        fig_box = px.box(
            dur_sample,
            x="attack_cat",
            y="dur",
            color="attack_cat",
            color_discrete_map=ATTACK_PALETTE,
            title="Flow Duration Distribution by Category",
            labels={"attack_cat": "Category", "dur": "Duration (s)"},
        )
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-30)
        plotly_dark_layout(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Summary table ──
    st.markdown('<p class="section-header">Dataset Summary Statistics</p>', unsafe_allow_html=True)
    summary_rows = []
    for cat, grp in df.groupby("attack_cat"):
        summary_rows.append({
            "Category":   cat,
            "Count":      len(grp),
            "% of Total": f"{len(grp)/total_records*100:.2f}%",
            "Avg Duration (s)": f"{grp['dur'].mean():.4f}",
            "Avg Total Bytes":  f"{grp['total_bytes'].mean():,.0f}",
            "Avg Rate (pps)":   f"{grp['rate'].mean():,.1f}",
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Count", ascending=False)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — Attack Type Analysis
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 Attack Type Analysis":
    st.markdown("# 🔍 Attack Type Analysis")
    st.markdown(
        "Deep-dive into individual attack categories, targeted protocols/services, "
        "traffic patterns, and flow characteristics."
    )
    st.markdown("---")

    # ── Filter: attack category selector ──
    all_cats = sorted(df["attack_cat"].unique().tolist())
    selected_cats = st.multiselect(
        "Filter by Attack Category (leave blank = all)",
        options=all_cats,
        default=[],
    )
    view_df = df[df["attack_cat"].isin(selected_cats)] if selected_cats else df

    # ── Row 1 ──
    st.markdown('<p class="section-header">Attack Category Deep-Dive</p>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        # Treemap
        tree_df = (
            view_df[view_df["label"] == 1]
            .groupby("attack_cat")
            .size()
            .reset_index(name="count")
        )
        fig_tree = px.treemap(
            tree_df,
            path=["attack_cat"],
            values="count",
            color="count",
            color_continuous_scale="YlOrRd",
            title="Attack Volume Treemap",
        )
        fig_tree.update_traces(textinfo="label+value+percent root")
        plotly_dark_layout(fig_tree)
        st.plotly_chart(fig_tree, use_container_width=True)

    with r1c2:
        # Sunburst: attack_cat → proto
        sun_df = (
            view_df[view_df["label"] == 1]
            .groupby(["attack_cat", "proto"])
            .size()
            .reset_index(name="count")
        )
        # Keep top 5 protocols per category
        top_p = df["proto"].value_counts().head(5).index
        sun_df = sun_df[sun_df["proto"].isin(top_p)]
        fig_sun = px.sunburst(
            sun_df,
            path=["attack_cat", "proto"],
            values="count",
            color="attack_cat",
            color_discrete_map=ATTACK_PALETTE,
            title="Attack Category → Protocol Breakdown",
        )
        plotly_dark_layout(fig_sun)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Row 2: Most-targeted protocols & services ──
    st.markdown('<p class="section-header">Most-Targeted Protocols & Services</p>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        proto_attack = (
            view_df[view_df["label"] == 1]["proto"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        proto_attack.columns = ["Protocol", "Count"]
        fig_p = px.bar(
            proto_attack,
            x="Count",
            y="Protocol",
            orientation="h",
            color="Count",
            color_continuous_scale="Oranges",
            title="Top 10 Protocols in Attack Traffic",
        )
        fig_p.update_layout(showlegend=False)
        plotly_dark_layout(fig_p)
        st.plotly_chart(fig_p, use_container_width=True)

    with r2c2:
        svc_attack = (
            view_df[view_df["label"] == 1]["service"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        svc_attack.columns = ["Service", "Count"]
        fig_s = px.bar(
            svc_attack,
            x="Count",
            y="Service",
            orientation="h",
            color="Count",
            color_continuous_scale="Reds",
            title="Top 10 Services in Attack Traffic",
        )
        fig_s.update_layout(showlegend=False)
        plotly_dark_layout(fig_s)
        st.plotly_chart(fig_s, use_container_width=True)

    # ── Row 3: Duration patterns ──
    st.markdown('<p class="section-header">Duration & Byte Patterns by Attack Type</p>', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        dur_cat = (
            view_df[view_df["label"] == 1]
            .groupby("attack_cat")["dur"]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        dur_cat.columns = ["attack_cat", "mean", "median", "std"]
        fig_dur = go.Figure()
        fig_dur.add_trace(
            go.Bar(
                name="Mean",
                x=dur_cat["attack_cat"],
                y=dur_cat["mean"],
                marker_color="#38bdf8",
            )
        )
        fig_dur.add_trace(
            go.Bar(
                name="Median",
                x=dur_cat["attack_cat"],
                y=dur_cat["median"],
                marker_color="#f97316",
            )
        )
        fig_dur.update_layout(
            title="Mean vs Median Duration per Attack Category",
            barmode="group",
            xaxis_tickangle=-30,
            xaxis_title="",
            yaxis_title="Duration (s)",
        )
        plotly_dark_layout(fig_dur)
        st.plotly_chart(fig_dur, use_container_width=True)

    with r3c2:
        byte_cat = (
            view_df[view_df["label"] == 1]
            .groupby("attack_cat")[["sbytes", "dbytes"]]
            .mean()
            .reset_index()
        )
        fig_bytes = go.Figure()
        fig_bytes.add_trace(
            go.Bar(
                name="Src Bytes (avg)",
                x=byte_cat["attack_cat"],
                y=byte_cat["sbytes"],
                marker_color="#a855f7",
            )
        )
        fig_bytes.add_trace(
            go.Bar(
                name="Dst Bytes (avg)",
                x=byte_cat["attack_cat"],
                y=byte_cat["dbytes"],
                marker_color="#22d3ee",
            )
        )
        fig_bytes.update_layout(
            title="Avg Source vs Destination Bytes per Category",
            barmode="stack",
            xaxis_tickangle=-30,
            xaxis_title="",
            yaxis_title="Bytes",
        )
        plotly_dark_layout(fig_bytes)
        st.plotly_chart(fig_bytes, use_container_width=True)

    # ── Row 4: Scatter & Heatmap ──
    st.markdown('<p class="section-header">Feature Relationships & State Heatmap</p>', unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        scatter_sample = (
            view_df[view_df["label"] == 1]
            .sample(min(3000, len(view_df[view_df["label"] == 1])), random_state=1)
        )
        fig_sc = px.scatter(
            scatter_sample,
            x="rate",
            y="dur",
            color="attack_cat",
            color_discrete_map=ATTACK_PALETTE,
            opacity=0.6,
            title="Rate vs Duration (Attack Flows)",
            labels={"rate": "Packet Rate (pps)", "dur": "Duration (s)"},
            size_max=6,
        )
        fig_sc.update_traces(marker=dict(size=4))
        plotly_dark_layout(fig_sc, height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

    with r4c2:
        # Heatmap: state × attack_cat
        heat_df = (
            view_df[view_df["label"] == 1]
            .groupby(["state", "attack_cat"])
            .size()
            .unstack(fill_value=0)
        )
        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale="YlOrRd",
            title="State × Attack Category Heatmap",
            labels=dict(x="Attack Category", y="State", color="Count"),
            aspect="auto",
        )
        plotly_dark_layout(fig_heat, height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Detailed stats table ──
    st.markdown('<p class="section-header">Detailed Attack Category Statistics</p>', unsafe_allow_html=True)
    detail_rows = []
    for cat, grp in attack_df.groupby("attack_cat"):
        detail_rows.append({
            "Category":          cat,
            "Flows":             len(grp),
            "% of Attacks":      f"{len(grp)/n_attacks*100:.2f}%",
            "Top Protocol":      grp["proto"].value_counts().idxmax(),
            "Top Service":       grp["service"].value_counts().idxmax(),
            "Top State":         grp["state"].value_counts().idxmax(),
            "Avg Dur (s)":       f"{grp['dur'].mean():.4f}",
            "Avg Rate (pps)":    f"{grp['rate'].mean():,.1f}",
            "Avg Src Bytes":     f"{grp['sbytes'].mean():,.0f}",
            "Avg Dst Bytes":     f"{grp['dbytes'].mean():,.0f}",
        })
    detail_df = pd.DataFrame(detail_rows).sort_values("Flows", ascending=False)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — Risk, Opportunity & Action
# ═══════════════════════════════════════════════════════════════════
elif page == "⚡ Risk, Opportunity & Action":
    st.markdown("# ⚡ Risk, Opportunity & Action")
    st.markdown(
        "Data-driven risk assessment, security improvement opportunities, "
        "and prioritised actionable recommendations."
    )
    st.markdown("---")

    # ── Risk scores (derived from data) ──
    st.markdown('<p class="section-header">🔴 Risk Assessment</p>', unsafe_allow_html=True)

    # Compute risk score = (flow_count / total_attacks * 50) + (avg_rate_normalised * 50)
    risk_data = []
    max_rate = attack_df.groupby("attack_cat")["rate"].mean().max()
    for cat, grp in attack_df.groupby("attack_cat"):
        flow_score = len(grp) / n_attacks * 50
        rate_score = (grp["rate"].mean() / max_rate) * 50 if max_rate > 0 else 0
        risk_data.append({
            "Category":  cat,
            "Flows":     len(grp),
            "Avg Rate":  grp["rate"].mean(),
            "Avg Bytes": grp["total_bytes"].mean(),
            "Risk Score": round(flow_score + rate_score, 1),
        })
    risk_df = pd.DataFrame(risk_data).sort_values("Risk Score", ascending=False)

    rg1, rg2 = st.columns(2)

    with rg1:
        fig_risk = px.bar(
            risk_df,
            x="Risk Score",
            y="Category",
            orientation="h",
            color="Risk Score",
            color_continuous_scale=["#22d3ee", "#f97316", "#ef4444"],
            title="Composite Risk Score by Attack Category",
            text="Risk Score",
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(showlegend=False)
        plotly_dark_layout(fig_risk)
        st.plotly_chart(fig_risk, use_container_width=True)

    with rg2:
        # Radar chart: multi-dimension per top attack
        radar_cats = risk_df.head(5)["Category"].tolist()
        metrics = ["Flows", "Avg Rate", "Avg Bytes", "Risk Score"]

        # Normalise 0-1
        radar_norm = risk_df[risk_df["Category"].isin(radar_cats)].copy()
        for m in metrics:
            _max = radar_norm[m].max()
            radar_norm[m] = radar_norm[m] / _max if _max > 0 else 0

        fig_radar = go.Figure()
        for _, row in radar_norm.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=row["Category"],
                    opacity=0.65,
                )
            )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0f172a",
                radialaxis=dict(visible=True, range=[0, 1], color="#64748b"),
                angularaxis=dict(color="#94a3b8"),
            ),
            title="Top-5 Attack Threat Radar",
        )
        plotly_dark_layout(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Protocol / Service Exposure ──
    st.markdown('<p class="section-header">🟡 Exposure Analysis</p>', unsafe_allow_html=True)
    eg1, eg2 = st.columns(2)

    with eg1:
        proto_exp = (
            attack_df.groupby("proto")
            .size()
            .reset_index(name="attack_flows")
        )
        proto_total = df.groupby("proto").size().reset_index(name="total_flows")
        proto_exp = proto_exp.merge(proto_total, on="proto")
        proto_exp["attack_ratio"] = proto_exp["attack_flows"] / proto_exp["total_flows"] * 100
        proto_exp = proto_exp.sort_values("attack_ratio", ascending=False).head(10)
        fig_exp = px.bar(
            proto_exp,
            x="attack_ratio",
            y="proto",
            orientation="h",
            color="attack_ratio",
            color_continuous_scale="YlOrRd",
            title="Protocol Attack Ratio (% of flows that are attacks)",
            text=proto_exp["attack_ratio"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_exp.update_traces(textposition="outside")
        fig_exp.update_layout(showlegend=False, xaxis_title="Attack %", yaxis_title="Protocol")
        plotly_dark_layout(fig_exp)
        st.plotly_chart(fig_exp, use_container_width=True)

    with eg2:
        svc_exp = (
            attack_df.groupby("service")
            .size()
            .reset_index(name="attack_flows")
        )
        svc_total = df.groupby("service").size().reset_index(name="total_flows")
        svc_exp = svc_exp.merge(svc_total, on="service")
        svc_exp["attack_ratio"] = svc_exp["attack_flows"] / svc_exp["total_flows"] * 100
        svc_exp = svc_exp.sort_values("attack_ratio", ascending=False).head(10)
        fig_svc = px.bar(
            svc_exp,
            x="attack_ratio",
            y="service",
            orientation="h",
            color="attack_ratio",
            color_continuous_scale="Reds",
            title="Service Attack Ratio (% of flows that are attacks)",
            text=svc_exp["attack_ratio"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_svc.update_traces(textposition="outside")
        fig_svc.update_layout(showlegend=False, xaxis_title="Attack %", yaxis_title="Service")
        plotly_dark_layout(fig_svc)
        st.plotly_chart(fig_svc, use_container_width=True)

    # ── Traffic anomaly indicators ──
    st.markdown('<p class="section-header">🟠 Traffic Anomaly Indicators</p>', unsafe_allow_html=True)
    ag1, ag2, ag3 = st.columns(3)

    high_rate_thresh = attack_df["rate"].quantile(0.95)
    high_rate_pct    = (attack_df["rate"] > high_rate_thresh).mean() * 100
    zero_dur_pct     = (attack_df["dur"] == 0).mean() * 100
    high_loss_pct    = (attack_df["sloss"] > 0).mean() * 100

    with ag1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#ef4444">{high_rate_pct:.1f}%</p>'
            f'<p class="kpi-label">High-Rate Attack Flows</p>'
            f'<p class="kpi-sub">Above 95th-pct threshold ({high_rate_thresh:,.0f} pps)</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with ag2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#eab308">{zero_dur_pct:.1f}%</p>'
            f'<p class="kpi-label">Zero-Duration Flows</p>'
            f'<p class="kpi-sub">Instant / scanning probes</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with ag3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:#f97316">{high_loss_pct:.1f}%</p>'
            f'<p class="kpi-label">Flows with Packet Loss</p>'
            f'<p class="kpi-sub">Source-side packet drops detected</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Cumulative flow trend ──
    st.markdown('<p class="section-header">📈 Cumulative Flow Volume (by Record ID)</p>', unsafe_allow_html=True)
    sample_trend = df.sort_values("id").assign(cum_attacks=lambda d: d["label"].cumsum())
    step = max(1, len(sample_trend) // 500)
    trend_plot = sample_trend.iloc[::step].copy()
    trend_plot["cum_total"] = range(step, len(sample_trend) + step, step)
    trend_plot["cum_total"] = trend_plot["id"]

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_plot["id"],
            y=trend_plot["cum_attacks"],
            mode="lines",
            name="Cumulative Attacks",
            line=dict(color="#ef4444", width=2),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)",
        )
    )
    fig_trend.update_layout(
        title="Cumulative Attack Flow Count vs Record ID",
        xaxis_title="Record ID",
        yaxis_title="Cumulative Attacks",
    )
    plotly_dark_layout(fig_trend, height=320)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Opportunities ──
    st.markdown("---")
    st.markdown('<p class="section-header">🟢 Security Improvement Opportunities</p>', unsafe_allow_html=True)

    opp_items = [
        ("Protocol-Level Filtering",
         f"**{proto_exp.iloc[0]['proto'].upper()}** carries the highest attack-to-normal ratio. "
         "Restrict or deeply inspect flows on this protocol at the perimeter.",
         "High"),
        ("Generic Attack Mitigation",
         "**Generic** attacks dominate (22.9% of all traffic). These typically exploit protocol "
         "weaknesses — deploying deep-packet inspection (DPI) and updating IDS signatures can reduce exposure.",
         "High"),
        ("Exploit Patch Management",
         "**Exploits** (11,132 flows) target known CVEs. An automated vulnerability scanner "
         "integrated with patch management can significantly reduce attack surface.",
         "High"),
        ("Fuzzer / Anomaly Detection",
         "**Fuzzers** generate malformed packets detectable by anomaly-based IDS. "
         "Enable statistical baseline thresholds for packet size and inter-arrival time.",
         "Medium"),
        ("DoS Rate Limiting",
         "**DoS** flows show elevated packet rates. Implement adaptive rate-limiting and "
         "traffic shaping at network ingress points.",
         "Medium"),
        ("Reconnaissance Detection",
         "**Reconnaissance** (3,496 flows) precedes targeted attacks. Monitor for port-scan "
         "patterns and short-duration probes with zero destination bytes.",
         "Medium"),
        ("Zero-Duration Flow Alerting",
         f"{zero_dur_pct:.1f}% of attack flows have zero duration — characteristic of scan probes. "
         "Flag and quarantine source IPs generating repeated zero-duration flows.",
         "Low"),
    ]

    priority_color = {"High": "badge-red", "Medium": "badge-yellow", "Low": "badge-green"}

    for title_o, desc, priority in opp_items:
        with st.expander(f"💡 {title_o}"):
            st.markdown(
                f'<span class="{priority_color[priority]}">{priority} Priority</span>',
                unsafe_allow_html=True,
            )
            st.markdown(desc)

    # ── Action Plan ──
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Prioritised Action Plan</p>', unsafe_allow_html=True)

    action_data = {
        "Priority": ["P1", "P1", "P1", "P2", "P2", "P3", "P3"],
        "Action": [
            "Update IDS/IPS signatures for Generic attack patterns",
            "Patch all systems against known Exploit CVEs (automated scanner)",
            "Block/throttle highest-attack-ratio protocols at perimeter",
            "Enable anomaly-based detection for Fuzzer patterns",
            "Deploy adaptive rate-limiting to counter DoS flows",
            "Implement port-scan detection for Reconnaissance flows",
            "Alert on zero-duration flows from repeated source IPs",
        ],
        "Expected Impact": [
            "Reduce Generic attacks by ~60%",
            "Reduce Exploit flows by ~70%",
            "Cut protocol-specific attack surface by ~40%",
            "Detect ~80% of fuzzing sessions",
            "Reduce DoS impact by ~50%",
            "Early-warn on ~90% of pre-attack reconnaissance",
            "Identify scanning hosts within minutes",
        ],
        "Effort": ["Medium", "High", "Low", "Medium", "Low", "Medium", "Low"],
        "Timeline": ["Week 1-2", "Week 1-4", "Day 1", "Week 2-3", "Day 1-3", "Week 2-3", "Day 1-3"],
    }
    action_df = pd.DataFrame(action_data)
    st.dataframe(action_df, use_container_width=True, hide_index=True)

    # ── Feature correlation heatmap ──
    st.markdown('<p class="section-header">🔬 Feature Correlation (Attack Flows)</p>', unsafe_allow_html=True)
    corr_cols = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sjit", "djit", "sload", "dload"]
    corr_cols = [c for c in corr_cols if c in attack_df.columns]
    corr_matrix = attack_df[corr_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Matrix (Attack Flows)",
        text_auto=".2f",
    )
    plotly_dark_layout(fig_corr, height=420)
    st.plotly_chart(fig_corr, use_container_width=True)
