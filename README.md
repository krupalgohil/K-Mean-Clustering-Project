# RetailLens — Customer Segmentation & Churn Intelligence

### End-to-End RFM Analysis & K-Means Clustering Pipeline

**Author:** Krupal Gohil &nbsp;|&nbsp; **Dataset:** UCI Online Retail II — 525K Transactions &nbsp;|&nbsp; **Domain:** E-Commerce / Customer Analytics

---

## Overview

Businesses that treat all customers the same leave money on the table. This project builds a complete customer intelligence pipeline on a real UK online gift retailer's transaction data — transforming **525,461 raw transaction rows** into **7 actionable customer segments** using RFM Analysis and K-Means Clustering.

Every customer receives a segment label and a churn risk score. The marketing team gets a prioritised action list instead of guessing who to contact, what offer to make, and how much budget to allocate.

> **32.9% of customers bought only once.** Moving even 10% of them to repeat buyers — without spending a single pound on new customer acquisition — is the direct business value this project enables.

---

## Segmentation Results

| Segment | Customers | Avg Spend (£) | Avg Orders | Profile |
|---------|----------:|-------------:|----------:|---------|
| DELIGHT | 14 | Very High | Very High | True VIP — both high spend and high frequency |
| UPSELL | 58 | 45,344 | 51.8 | Extreme frequency — buy constantly, lower basket size |
| REWARD | 845 | 4,253 | 9.9 | Best core customers — recent, frequent, high value |
| RETAIN | 1,396 | 1,248 | 3.5 | Stable mid-tier — reliable and consistent |
| NURTURE | 1,108 | 317 | 1.4 | New or one-time buyers — highest growth potential |
| RE-ENGAGE | 864 | 409 | 1.4 | Lapsed customers — 254 days since last purchase |
| **Total** | **4,285** | | | **7 segments across 4,285 unique customers** |

---

## Project Structure

```
RETAILLENS/
│
├── Visuals/
│   ├── Top 10 Countries by Revenue.png
│   ├── Monthly Revenue Trend (Dec 2009 – Dec 2010).png
│   ├── Transaction amount distribution.png
│   ├── No of order per custome.png
│   └── Days Since Last Purchase — All Customers.png
│
├── online_retail_II.xlsx          ← Dataset (not pushed to GitHub)
├── Customer_Segmentation.ipynb    ← Full analysis notebook (19 steps)
├── app.py                         ← Streamlit dashboard
├── requirements.txt               ← Dependencies
├── .gitignore
└── README.md
```

---

## Dataset

**UCI Online Retail II — Real Transaction Data**

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- Real transactions from a UK-based online gift retailer
- Period: December 2009 – December 2010 (13 months)
- 525,461 rows across 40 countries

| Column | Description |
|--------|-------------|
| `Invoice` | Unique invoice number — prefix `C` = cancellation, `A` = bad debt |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units sold — negative values are returns |
| `InvoiceDate` | Date and time of transaction |
| `Price` | Unit price in GBP (£) |
| `Customer ID` | Unique customer identifier — nullable (guest checkouts) |
| `Country` | Customer country |

---

## Exploratory Data Analysis

Each chart below answers a specific business question.

---

### 1. Which countries generate the most revenue?

![Top 10 Countries by Revenue](Visuals/Top%2010%20Countries%20by%20Revenue.png)

The UK accounts for **87.8% of total revenue** — this is fundamentally a UK domestic business with some international presence. Netherlands, EIRE, Germany, and France make up most of the remaining 12.2%. This tells us where to focus retention spend: UK customer loyalty is the single most important lever for revenue growth.

---

### 2. How does revenue change over time?

![Monthly Revenue Trend](Visuals/Monthly%20Revenue%20Trend%20(Dec%202009%20%E2%80%93%20Dec%202010).png)

Revenue follows a clear seasonal pattern with a significant spike in **Q4 (October–November)** driven by Christmas gift purchasing. Customers who bought only during this peak period may look high-value but are actually seasonal one-off buyers. The NURTURE segment captures many of these customers.

---

### 3. What does the transaction amount distribution look like?

![Transaction Amount Distribution](Visuals/Transaction%20amount%20distribution.png)

The distribution is **heavily right-skewed** — the median transaction is just **£10.14**, the mean is £20.15, and the 90th percentile is only £33.90. A small number of large wholesale orders pull the average up significantly. This is the classic 80/20 pattern that makes segmentation so valuable: a handful of customers drive a disproportionate share of revenue.

---

### 4. How often do customers typically order?

![Orders per Customer](Visuals/No%20of%20order%20per%20custome.png)

**32.9% of customers (1,419 out of 4,314) bought only once.** The median customer placed just 2 orders total. This is the single most actionable insight in the project — converting even 10% of one-time buyers to repeat customers adds ~142 loyal customers at zero acquisition cost.

---

### 5. How recently did customers last purchase?

![Days Since Last Purchase](Visuals/Days%20Since%20Last%20Purchase%20%E2%80%94%20All%20Customers.png)

There are two distinct groups in the Recency distribution — recently active customers and a long tail of lapsed buyers. **1,427 customers have not purchased in 90+ days** and **825 have not purchased in 180+ days**. These customers form the RE-ENGAGE segment and are the primary targets for win-back campaigns.

---

## RFM Feature Engineering

RFM is an industry-standard customer scoring framework used in CRM systems worldwide. We compute three metrics for each of the 4,285 customers.

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| **Recency** | Days since last purchase (from snapshot date Dec 2010) | Lower = more recently active = better |
| **Frequency** | Count of unique invoice numbers | Higher = more loyal = better |
| **Monetary** | Sum of (Price × Quantity) across all purchases | Higher = more valuable = better |

### Why Log Transformation?

Both `MonetaryValue` and `Frequency` are heavily right-skewed. Without transformation, K-Means treats a £50,000 spender as being 5,000× farther away from a £10 customer than the data warrants. Applying `np.log1p()` compresses the long tail and gives the algorithm a fair view of relative customer differences.

---

## Outlier Handling

72 customers (1.7%) had extreme Monetary or Frequency values even after log transformation. Including them in K-Means would pull cluster centroids away from the 4,213 regular customers.

**We do not drop them — they are the most valuable customers in the entire dataset.** Instead, we separate them into dedicated premium segments before clustering.

| Premium Segment | Customers | Criterion |
|----------------|----------:|-----------|
| DELIGHT | 14 | Both high Monetary AND high Frequency — true VIPs |
| UPSELL | 58 | High Frequency only — buy very often, lower basket |
| PAMPER | — | High Monetary only (no customers met this threshold) |

---

## Clustering

### Finding the Optimal K

We tested K from 2 to 10 using two complementary methods:

- **Elbow Method (Inertia)** — measures cluster compactness; look for where the curve bends
- **Silhouette Score** — measures cluster separation; higher is better (range: −1 to +1)

The silhouette score peaks at k=3 (score: **0.4146**). We chose **k=4** because the elbow bends clearly at 4 clusters, and four segments give more distinct, business-actionable groups than three. In real CRM projects, business interpretability alongside statistical metrics guides the final choice.

### Stability Check

K-Means was run 10 times with different random seeds. Low standard deviation in silhouette scores across runs confirms the clusters are stable — real patterns in the data, not a product of lucky initialisation.

### Cluster Profiles (Original RFM Values)

| Cluster | Avg Recency | Avg Frequency | Avg Monetary | Label Assigned |
|---------|------------:|--------------:|-------------:|---------------|
| 0 | 254.6 days | 1.4 orders | £409 | RE-ENGAGE |
| 1 | 54.7 days | 3.5 orders | £1,248 | RETAIN |
| 2 | 25.8 days | 9.9 orders | £4,253 | REWARD |
| 3 | 55.6 days | 1.4 orders | £317 | NURTURE |

---

## Churn Risk Scoring

Every customer receives a churn risk score from 0–100 based on normalised Recency.

| Score Range | Risk Level | Recommended Action |
|-------------|-----------|-------------------|
| 0–30 | Low | Monitor — continue regular communication |
| 30–60 | Medium | Soft nudge — seasonal promotion or product update |
| 60–100 | High | Win-back campaign — immediate priority |

The RE-ENGAGE segment carries the highest churn scores (avg Recency of 254 days). Critically, a hidden subset of RETAIN customers also have elevated scores — drifting quietly without triggering any rule-based alert. This is exactly the kind of pattern that machine learning surfaces that manual rules cannot.

---

## Business Recommendations

### Segment-by-Segment Action Plan

| Segment | Priority | Key Action |
|---------|----------|-----------|
| DELIGHT | Critical | VIP programme, dedicated account contact, early product access |
| UPSELL | High | Cross-sell higher-margin products, volume discount thresholds |
| REWARD | High | Points-based loyalty scheme, referral programme |
| RETAIN | Medium | Regular newsletters, seasonal promotions |
| NURTURE | Medium | Welcome email sequence, first-repeat-purchase incentive |
| RE-ENGAGE | Targeted | Win-back campaign: "We miss you — here's 20% off" |

### Budget Allocation

| Priority Group | Segments | Suggested Share |
|----------------|---------|----------------|
| Protect & grow | DELIGHT + UPSELL + REWARD | 50% |
| Develop | RETAIN + NURTURE | 35% |
| Recover | RE-ENGAGE | 15% |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/krupalgohil/RetailLens.git
cd RetailLens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `online_retail_II.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in the project root folder.

### 4. Run the notebook

Open `Customer_Segmentation.ipynb` in Jupyter and run all cells top to bottom. All 19 steps execute sequentially with no manual intervention required.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas, NumPy | Data cleaning, RFM aggregation |
| Seaborn, Matplotlib | All visualisations (EDA + model results) |
| Scikit-Learn | K-Means, StandardScaler, PCA, Silhouette Score |
| OpenPyXL | Reading the Excel dataset |

---

## Business Impact

For a retailer with ~4,285 known customers:

- **DELIGHT + UPSELL (72 customers)** generate a disproportionate share of total revenue — losing 5 of them has the same impact as losing hundreds of NURTURE customers combined
- **RE-ENGAGE win-back campaign** — 20% response rate × £500 average order × 864 customers = **£86,400 in recovered revenue** from a single targeted email
- **NURTURE → RETAIN conversion** — moving 10% of one-time buyers to repeat customers adds ~142 recurring customers at zero acquisition cost
- **Total time saved** — the full segmentation pipeline runs in minutes; manual customer analysis of this dataset would take days

---

## Future Improvements

- Monthly automated re-scoring pipeline to track how customers migrate between segments over time
- RFM quintile scoring (1–5 per dimension) for standard CRM tool integration
- Predictive churn classifier to estimate probability of churning in the next 30 days
- Product affinity analysis per segment for truly personalised product recommendations
- A/B test framework to measure whether segment-targeted campaigns outperform generic sends
- REST API using FastAPI for real-time customer scoring at checkout

---

## References

1. Chen, D. et al. (2012). *Online Retail II Data Set.* UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii
