# Customer Intelligence & Segmentation Engine

**RFM Analysis + K-Means Clustering on 500K+ real e-commerce transactions**

---

## What this project does

Takes raw transactional data from an online gift retailer and segments ~4,300 customers into 7 actionable groups using K-Means clustering. Each segment gets a clear business label and a recommended marketing strategy.

The output is a ranked, labelled customer list that a marketing team can import into any CRM tool.

---

## Segments discovered

| Segment | Description |
|---------|-------------|
| DELIGHT | True VIPs — high spend, high frequency, recently active |
| PAMPER | High-ticket buyers — big orders, less frequent |
| UPSELL | Frequent buyers — buy often but low basket size |
| REWARD | Active loyalists — consistent and recently active |
| RETAIN | Stable mid-tier — steady but need engagement |
| NURTURE | New or occasional buyers — high growth potential |
| RE-ENGAGE | Lapsed customers — haven't purchased in a while, churn risk |

---

## Project structure

```
├── Customer_Intelligence_Segmentation.ipynb   # Main analysis notebook
├── app.py                                      # Streamlit dashboard
├── requirements.txt                            # Python dependencies
└── online_retail_II.xlsx                       # Dataset (download separately)
```

---

## How to run

### 1. Get the dataset
Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place `online_retail_II.xlsx` in the project folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Customer_Intelligence_Segmentation.ipynb` in Jupyter and run all cells.

### 4. Launch the dashboard (optional)
```bash
streamlit run app.py
```

---

## Tech stack

- **Python 3.10+**
- **Pandas** — data cleaning and RFM aggregation
- **Scikit-Learn** — K-Means, StandardScaler, PCA, Silhouette Score
- **Matplotlib / Seaborn** — visualisations
- **Streamlit** — interactive dashboard

---

## Dataset

- **Source:** UCI Machine Learning Repository — Online Retail II
- **Period:** December 2009 – December 2010
- **Size:** ~525,000 transactions across 40 countries
- **Link:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
