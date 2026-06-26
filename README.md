👥 RFM Customer Segmentation — LPG Distribution Accounts
> **Python-based customer segmentation using RFM analysis and K-Means clustering to classify 250 distribution accounts into actionable segments for priority management.**
🎯 Objective
Segment 250 accounts (25 retailers + 225 individual buyers) based on purchase behavior to prioritize delivery resources, identify at-risk accounts, and optimize reorder allocation.
🛠️ Tools
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`
🔍 Methodology
RFM Calculation — Recency, Frequency, Monetary per account
Quintile Scoring — R/F/M scores 1–5
K-Means Clustering — Optimal k=4 via Elbow + Silhouette
Segment Labeling — Champions / Loyal / At Risk / Lost
📊 Visualizations
Elbow method + Silhouette score (cluster selection)
Segment distribution pie + normalized RFM bar chart
Frequency vs Monetary scatter by segment
RFM score heatmap by segment
📁 Structure
```
lgp-rfm-segmentation/
├── rfm_customer_segmentation.py
├── README.md
└── outputs/
    ├── plot_rfm_elbow.png
    ├── plot_rfm_segments.png
    ├── plot_rfm_scatter.png
    └── plot_rfm_heatmap.png
```
🚀 How to Run
```bash
# Google Colab: paste cell by cell and run
# Local:
pip install pandas numpy matplotlib seaborn scikit-learn
python rfm_customer_segmentation.py
```
📌 CV Bullet (Ops/Analytics CV — Analytical Projects)
> **RFM Customer Segmentation — LPG Distribution Accounts** | Python, Scikit-learn, Seaborn | 2026
> Segmented 250 distribution accounts (retailers + individual buyers) using RFM analysis and K-Means clustering (k=4, validated via Silhouette score); identified Champions, Loyal, At-Risk, and Inactive segments to support priority delivery scheduling and account reactivation strategy.
👤 Author
Ferdy Febrian Iskandar · iskandarferdy559@gmail.com
Dataset simulated based on real operational experience. For portfolio purposes only.
