# ============================================================
# RFM Customer Segmentation — LPG Distribution Accounts
# Pertamina-Authorized LPG 3kg Sub-Agent | Tanah Datar, West Sumatra
# Author: Ferdy Febrian Iskandar
# Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
# ============================================================

# CELL 1: Install & Import
# !pip install pandas numpy matplotlib seaborn scikit-learn  # uncomment if needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("✅ Libraries loaded.")

# CELL 2: Generate Transaction Dataset
# 250 accounts (25 retailers + 225 individual buyers)
# Period: Dec 2024 – May 2026

MONTHS = pd.date_range(start='2024-12-01', end='2026-05-01', freq='MS')
RETAILERS = [f'Toko_{i:02d}' for i in range(1, 26)]
BUYERS    = [f'Buyer_{i:03d}' for i in range(1, 226)]
ALL_ACCOUNTS = RETAILERS + BUYERS

ACCOUNT_TYPE = {a: 'Retailer' for a in RETAILERS}
ACCOUNT_TYPE.update({a: 'Individual' for a in BUYERS})

# Retailer: order 40–120 cyl/month, high frequency
# Individual buyer: order 1–3 cyl, irregular frequency
transactions = []
for account in ALL_ACCOUNTS:
    is_retailer = account in RETAILERS
    active_months = np.random.choice(MONTHS, size=np.random.randint(
        12 if is_retailer else 3, len(MONTHS)+1 if is_retailer else 12
    ), replace=False)
    for month in active_months:
        qty = int(np.random.normal(75, 20)) if is_retailer else np.random.choice([1,2,3], p=[0.6,0.3,0.1])
        qty = max(1, qty)
        day = np.random.randint(1, 28)
        date = month + pd.Timedelta(days=day)
        transactions.append({'account_id': account, 'type': ACCOUNT_TYPE[account],
                              'date': date, 'qty': qty})

tx = pd.DataFrame(transactions)
tx['date'] = pd.to_datetime(tx['date'])
print(f"✅ {len(tx):,} transactions | {tx['account_id'].nunique()} unique accounts")

# CELL 3: Calculate RFM Scores
# Reference date: one day after last transaction
ref_date = tx['date'].max() + pd.Timedelta(days=1)

rfm = tx.groupby('account_id').agg(
    Recency   = ('date', lambda x: (ref_date - x.max()).days),
    Frequency = ('date', 'count'),
    Monetary  = ('qty', 'sum')
).reset_index()
rfm['type'] = rfm['account_id'].map(ACCOUNT_TYPE)

print("\n📊 RFM Summary:")
print(rfm[['Recency','Frequency','Monetary']].describe().round(1))

# CELL 4: RFM Scoring (1–5 quintiles)
rfm['R_score'] = pd.qcut(rfm['Recency'],   5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# CELL 5: K-Means Clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

# Optimal k=4
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

# Label clusters based on RFM_total median
cluster_summary = rfm.groupby('Cluster')[['Recency','Frequency','Monetary','RFM_total']].mean()
rank_order = cluster_summary['RFM_total'].rank(ascending=False).astype(int)
label_map = {c: ['Champions','Loyal Customers','At Risk','Lost/Inactive'][r-1]
             for c, r in rank_order.items()}
rfm['Segment'] = rfm['Cluster'].map(label_map)

print("\n📋 Cluster Profiles:")
print(rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1))

# CELL 6: Plot 1 — Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'o-', color='#1565C0', linewidth=2)
axes[0].axvline(4, color='red', linestyle='--', linewidth=1, label='Optimal k=4')
axes[0].set_title('Elbow Method — Optimal K', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].spines[['top','right']].set_visible(False)

axes[1].plot(K_range, sil_scores, 's-', color='#43A047', linewidth=2)
axes[1].axvline(4, color='red', linestyle='--', linewidth=1, label='Optimal k=4')
axes[1].set_title('Silhouette Score by K', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].spines[['top','right']].set_visible(False)

plt.suptitle('K-Means Optimal Cluster Selection', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_rfm_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_rfm_elbow.png")

# CELL 7: Plot 2 — Segment Distribution
seg_counts = rfm['Segment'].value_counts()
seg_colors = {'Champions':'#1565C0','Loyal Customers':'#43A047',
              'At Risk':'#FB8C00','Lost/Inactive':'#E53935'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = [seg_colors[s] for s in seg_counts.index]
axes[0].pie(seg_counts, labels=seg_counts.index, colors=colors, autopct='%1.1f%%',
            startangle=140, textprops={'fontsize':10})
axes[0].set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')

seg_rfm = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().reindex(
    ['Champions','Loyal Customers','At Risk','Lost/Inactive'])
x = np.arange(len(seg_rfm))
w = 0.25
axes[1].bar(x-w, seg_rfm['Recency']/seg_rfm['Recency'].max()*100,   w, label='Recency (inv)', color='#1565C0', alpha=0.8)
axes[1].bar(x,   seg_rfm['Frequency']/seg_rfm['Frequency'].max()*100, w, label='Frequency',    color='#43A047', alpha=0.8)
axes[1].bar(x+w, seg_rfm['Monetary']/seg_rfm['Monetary'].max()*100,  w, label='Monetary',     color='#FB8C00', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(seg_rfm.index, rotation=12, fontsize=9)
axes[1].set_title('RFM Profile by Segment (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Normalized Score (%)')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('plot_rfm_segments.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_rfm_segments.png")

# CELL 8: Plot 3 — Scatter RFM (Frequency vs Monetary, color by segment)
fig, ax = plt.subplots(figsize=(11, 7))
for seg, color in seg_colors.items():
    subset = rfm[rfm['Segment']==seg]
    ax.scatter(subset['Frequency'], subset['Monetary'], c=color, label=seg,
               alpha=0.6, s=60, edgecolors='white', linewidths=0.4)
ax.set_title('RFM Scatter — Frequency vs Monetary Value by Segment', fontsize=13, fontweight='bold')
ax.set_xlabel('Frequency (Number of Orders)', fontsize=10)
ax.set_ylabel('Monetary (Total Cylinders)', fontsize=10)
ax.legend(title='Segment', fontsize=9)
ax.grid(alpha=0.25)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('plot_rfm_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_rfm_scatter.png")

# CELL 9: Plot 4 — Heatmap RFM by Segment
fig, ax = plt.subplots(figsize=(8, 4))
heat_data = rfm.groupby('Segment')[['R_score','F_score','M_score']].mean().reindex(
    ['Champions','Loyal Customers','At Risk','Lost/Inactive']).round(1)
sns.heatmap(heat_data, annot=True, fmt='.1f', cmap='YlGn', linewidths=0.5,
            linecolor='white', ax=ax, cbar_kws={'label':'Avg Score'})
ax.set_title('RFM Score Heatmap by Customer Segment', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_rfm_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_rfm_heatmap.png")

# CELL 10: Actionable Insights
print("\n" + "="*60)
print("💡 SEGMENT INSIGHTS & RECOMMENDED ACTIONS")
print("="*60)
for seg in ['Champions','Loyal Customers','At Risk','Lost/Inactive']:
    subset = rfm[rfm['Segment']==seg]
    print(f"\n{seg} ({len(subset)} accounts):")
    print(f"  Avg Recency  : {subset['Recency'].mean():.0f} days")
    print(f"  Avg Frequency: {subset['Frequency'].mean():.1f} orders")
    print(f"  Avg Monetary : {subset['Monetary'].mean():.0f} cylinders")

actions = {
    'Champions':       'Prioritize for fast delivery SLA; offer loyalty pricing',
    'Loyal Customers': 'Maintain regular contact; upsell larger order packages',
    'At Risk':         'Re-engage with proactive outreach; investigate drop-off cause',
    'Lost/Inactive':   'Assess if worth re-activation; reallocate quota to active accounts'
}
print("\n📌 Recommended Actions:")
for seg, action in actions.items():
    print(f"  {seg}: {action}")

print("\n✅ RFM Segmentation complete.")
