"""
Customer Intelligence & Segmentation Dashboard
Run with: streamlit run app.py
Requires: online_retail_II.xlsx in the same folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Customer Intelligence & Segmentation")
st.markdown("**RFM Analysis + K-Means Clustering** on Online Retail II Dataset")
st.divider()

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df = pd.read_excel("online_retail_II.xlsx", sheet_name=0)

    # Clean
    df["Invoice"] = df["Invoice"].astype("str")
    df["StockCode"] = df["StockCode"].astype("str")

    valid_invoice = df["Invoice"].str.match(r"^\d{6}$")
    valid_stock = (
        df["StockCode"].str.match(r"^\d{5}$") |
        df["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$") |
        (df["StockCode"] == "PADS")
    )
    cleaned = df[valid_invoice & valid_stock].copy()
    cleaned.dropna(subset=["Customer ID"], inplace=True)
    cleaned = cleaned[cleaned["Price"] > 0]

    # RFM
    cleaned["SalesLineTotal"] = cleaned["Price"] * cleaned["Quantity"]
    rfm = cleaned.groupby("Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastPurchase=("InvoiceDate", "max")
    )
    snapshot = rfm["LastPurchase"].max()
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
    rfm.drop(columns="LastPurchase", inplace=True)

    # Log transform
    rfm_log = rfm.copy()
    rfm_log["MonetaryValue"] = np.log1p(rfm_log["MonetaryValue"])
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])

    # Outliers
    def iqr_mask(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    out_mask = iqr_mask(rfm_log["MonetaryValue"]) | iqr_mask(rfm_log["Frequency"])
    core_log = rfm_log[~out_mask]
    outlier_rfm = rfm[out_mask].copy()

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(core_log[["MonetaryValue", "Frequency", "Recency"]])
    scaled_df = pd.DataFrame(scaled, index=core_log.index, columns=["MonetaryValue", "Frequency", "Recency"])

    # K-Means (k=4)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    core_rfm = rfm[~out_mask].copy()
    core_rfm["Cluster"] = km.fit_predict(scaled_df)

    segment_map = {0: "RETAIN", 1: "RE-ENGAGE", 2: "NURTURE", 3: "REWARD"}
    core_rfm["Segment"] = core_rfm["Cluster"].map(segment_map)

    # Outlier segments
    mon_out = iqr_mask(rfm_log.loc[out_mask, "MonetaryValue"])
    freq_out = iqr_mask(rfm_log.loc[out_mask, "Frequency"])

    def premium_label(idx):
        m = mon_out.get(idx, False)
        f = freq_out.get(idx, False)
        if m and f:
            return "DELIGHT"
        elif m:
            return "PAMPER"
        return "UPSELL"

    outlier_rfm["Segment"] = [premium_label(i) for i in outlier_rfm.index]
    outlier_rfm["Cluster"] = -1

    # Merge
    all_seg = pd.concat([
        core_rfm[["Customer ID", "Recency", "Frequency", "MonetaryValue", "Segment"]],
        outlier_rfm[["Customer ID", "Recency", "Frequency", "MonetaryValue", "Segment"]]
    ])
    max_rec = all_seg["Recency"].max()
    all_seg["ChurnScore"] = (all_seg["Recency"] / max_rec * 100).round(1)

    # PCA for scatter
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=core_rfm.index)
    pca_df["Segment"] = core_rfm["Segment"].values

    return all_seg, pca_df, pca.explained_variance_ratio_, df


# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Loading and processing data..."):
    all_seg, pca_df, var_ratio, raw_df = load_and_process()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
all_segs = sorted(all_seg["Segment"].unique())
selected = st.sidebar.multiselect("Show segments", all_segs, default=all_segs)
filtered = all_seg[all_seg["Segment"].isin(selected)]

st.sidebar.divider()
st.sidebar.markdown("**About this project**")
st.sidebar.markdown("RFM segmentation on UCI Online Retail II dataset using K-Means clustering.")
st.sidebar.markdown("**7 segments** · ~4,300 customers · £500K+ revenue analysed")

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(all_seg):,}")
col2.metric("Segments", "7")
col3.metric("Avg Monetary Value", f"£{all_seg['MonetaryValue'].mean():,.0f}")
col4.metric("High Churn Risk", f"{(all_seg['ChurnScore'] > 70).sum():,}")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Segment Overview", "EDA", "Cluster Visualisation", "At-Risk Customers"])

# ── Tab 1: Segment Overview ───────────────────────────────────────────────────
with tab1:
    st.subheader("Segment Profiles")

    summary = filtered.groupby("Segment").agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("MonetaryValue", "mean"),
        Avg_ChurnScore=("ChurnScore", "mean")
    ).round(1).sort_values("Avg_Monetary", ascending=False)

    st.dataframe(summary, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        summary["Customers"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Customers per Segment")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Heatmap
        heat = filtered.groupby("Segment")[["Recency", "Frequency", "MonetaryValue"]].mean()
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min())
        sns.heatmap(heat_norm, annot=heat.round(0), fmt=".0f",
                    cmap="RdYlGn_r", ax=ax, linewidths=0.5)
        ax.set_title("RFM Heatmap (normalised)")
        st.pyplot(fig)
        plt.close()

# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    eda = raw_df.copy()
    eda["Revenue"] = eda["Quantity"] * eda["Price"]
    eda_pos = eda[eda["Revenue"] > 0]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 10 Countries by Revenue**")
        top_c = eda_pos.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_c.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Monthly Revenue Trend**")
        eda_pos = eda_pos.copy()
        eda_pos["Month"] = pd.to_datetime(eda_pos["InvoiceDate"]).dt.to_period("M")
        monthly = eda_pos.groupby("Month")["Revenue"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly.plot(ax=ax, color="teal", marker="o", linewidth=2)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close()

    st.markdown("**Top 10 Products by Revenue**")
    top_p = eda_pos.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 3))
    top_p.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Revenue (£)")
    st.pyplot(fig)
    plt.close()

# ── Tab 3: Cluster Visualisation ─────────────────────────────────────────────
with tab3:
    st.subheader("PCA 2D Cluster Scatter")
    st.caption(f"PC1 explains {var_ratio[0]*100:.1f}% variance · PC2 explains {var_ratio[1]*100:.1f}% variance")

    seg_palette = {
        "RETAIN": "#1f77b4", "RE-ENGAGE": "#ff7f0e",
        "NURTURE": "#2ca02c", "REWARD": "#d62728"
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for seg, color in seg_palette.items():
        mask = pca_df["Segment"] == seg
        ax.scatter(pca_df[mask]["PC1"], pca_df[mask]["PC2"],
                   c=color, label=seg, alpha=0.5, s=15)
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Customer Clusters in Reduced RFM Space")
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("RFM Distribution by Segment")
    metric = st.selectbox("Feature", ["Recency", "Frequency", "MonetaryValue"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x="Segment", y=metric, data=filtered,
                order=[s for s in all_segs if s in filtered["Segment"].unique()],
                palette="Set2", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close()

# ── Tab 4: At-Risk Customers ──────────────────────────────────────────────────
with tab4:
    st.subheader("Churn Risk — RE-ENGAGE Customers")
    st.markdown("Customers ranked by churn risk score (100 = highest risk). Export this list for win-back campaigns.")

    at_risk = (
        all_seg[all_seg["Segment"] == "RE-ENGAGE"]
        .sort_values("ChurnScore", ascending=False)
        .reset_index(drop=True)
        [["Customer ID", "Recency", "Frequency", "MonetaryValue", "ChurnScore"]]
    )
    at_risk.columns = ["Customer ID", "Days Since Purchase", "Orders", "Total Spend (£)", "Churn Risk Score"]

    st.dataframe(at_risk, use_container_width=True)

    csv = at_risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download at-risk customer list (CSV)",
        data=csv,
        file_name="at_risk_customers.csv",
        mime="text/csv"
    )
