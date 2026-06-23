import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
import difflib

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("--- PHASE 1: DATA UNDERSTANDING ---")
df_raw = pd.read_csv('data/online_retail.csv')
print(f"Raw Dataset Shape: {df_raw.shape}")
print("\nUnique Customers:", df_raw['CustomerID'].nunique())
print("Unique Products:", df_raw['Description'].nunique())

print("\n--- PHASE 2: DATA CLEANING ---")
# Cleaning steps
df_clean = df_raw.dropna(subset=['CustomerID'])
df_clean = df_clean.dropna(subset=['Description'])
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]
df_clean = df_clean.drop_duplicates()
df_clean['Description'] = df_clean['Description'].str.strip()
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

print(f"Cleaned Dataset Shape: {df_clean.shape}")
print(f"Data Retained: {len(df_clean)/len(df_raw)*100:.2f}%")

print("\n--- PHASE 4: RFM FEATURE ENGINEERING ---")
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Snapshot Date: {snapshot_date}")

# Calculate RFM
rfm_df = df_clean.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'), # unique InvoiceNo count
    Monetary=('TotalAmount', 'sum')
)
print("RFM stats:")
print(rfm_df.describe())

print("\n--- PHASE 5: FEATURE SCALING ---")
# Apply log1p to Frequency and Monetary
rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency'])
rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary'])
rfm_df['Recency_log'] = rfm_df['Recency'] # Keep Recency as is

# Scale
features_to_scale = ['Recency', 'Frequency_log', 'Monetary_log']
scaler = StandardScaler()
rfm_scaled_arr = scaler.fit_transform(rfm_df[features_to_scale])
rfm_scaled_df = pd.DataFrame(rfm_scaled_arr, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'], index=rfm_df.index)

print("\n--- PHASE 6: CLUSTER SELECTION ---")
inertia_values = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled_arr)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled_arr, labels))

eval_summary = pd.DataFrame({
    'K Value': list(k_range),
    'Inertia': inertia_values,
    'Silhouette Score': silhouette_scores
})
print("Cluster Evaluation Summary Table:")
print(eval_summary.to_string(index=False))

print("\n--- PHASE 7: CUSTOMER SEGMENTATION ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled_arr)

# Programmatic Mapping based on centroid statistics
cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
high_value_cluster = cluster_means['Monetary'].idxmax()

remaining_clusters = [c for c in cluster_means.index if c != high_value_cluster]
at_risk_cluster = cluster_means.loc[remaining_clusters, 'Recency'].idxmax()

remaining_clusters = [c for c in remaining_clusters if c != at_risk_cluster]
if cluster_means.loc[remaining_clusters[0], 'Monetary'] > cluster_means.loc[remaining_clusters[1], 'Monetary']:
    regular_cluster = remaining_clusters[0]
    occasional_cluster = remaining_clusters[1]
else:
    regular_cluster = remaining_clusters[1]
    occasional_cluster = remaining_clusters[0]

segment_mapping = {
    high_value_cluster: 'High Value',
    regular_cluster: 'Regular',
    occasional_cluster: 'Occasional',
    at_risk_cluster: 'At Risk'
}
rfm_df['Segment'] = rfm_df['Cluster'].map(segment_mapping)

print("Segment Customer Counts:")
print(rfm_df['Segment'].value_counts())

# Generate customer segment summary table
total_rev = rfm_df['Monetary'].sum()
segment_summary = rfm_df.groupby('Segment').agg(
    Customer_Count=('Recency', 'count'),
    Revenue_Contribution_Pct=('Monetary', lambda x: (x.sum() / total_rev) * 100),
    Recency_Avg=('Recency', 'mean'),
    Frequency_Avg=('Frequency', 'mean'),
    Monetary_Avg=('Monetary', 'mean')
).reset_index()

print("\nCustomer Segment Summary Table:")
print(segment_summary.to_string(index=False))

print("\n--- PHASE 9: BUSINESS INSIGHTS ---")
segment_financials = rfm_df.groupby('Segment').agg(
    Customer_Count=('Recency', 'count'),
    Total_Revenue=('Monetary', 'sum')
).reset_index()
segment_financials['Revenue_Pct'] = (segment_financials['Total_Revenue'] / total_rev) * 100
print(segment_financials.to_string(index=False))

print("\n--- PHASE 10: PRODUCT RECOMMENDATION SYSTEM ---")
# Count unique customers per product
product_cust_counts = df_clean.groupby('Description')['CustomerID'].nunique()
# Filter out products with < 5 purchases
min_purchases = 5
popular_products = product_cust_counts[product_cust_counts >= min_purchases].index
df_filtered_recs = df_clean[df_clean['Description'].isin(popular_products)]
print(f"Filtered products count: {len(popular_products)} (from {len(product_cust_counts)})")

# Create customer product matrix
customer_product_matrix = df_filtered_recs.pivot_table(
    index='CustomerID', 
    columns='Description', 
    values='Quantity', 
    aggfunc='sum'
).fillna(0)

# Product similarity
product_similarity = cosine_similarity(customer_product_matrix.T)
product_similarity_df = pd.DataFrame(
    product_similarity, 
    index=customer_product_matrix.columns, 
    columns=customer_product_matrix.columns
)

print("\n--- PHASE 12: SAVE MODELS AND METRICS ---")
os.makedirs('models', exist_ok=True)
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(product_similarity_df, 'models/product_similarity.pkl')
joblib.dump(list(product_similarity_df.index), 'models/product_names.pkl')

# Calculate monthly trend data
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
monthly_data = df_clean.groupby('YearMonth').agg(
    Revenue=('TotalAmount', 'sum'),
    Transactions=('InvoiceNo', 'nunique')
).reset_index()
monthly_data['YearMonth_str'] = monthly_data['YearMonth'].astype(str)

# Calculate top 5 revenue products
top_5_products = df_clean.groupby('Description').agg(
    Revenue=('TotalAmount', 'sum'),
    Quantity=('Quantity', 'sum')
).sort_values(by='Revenue', ascending=False).head(5).reset_index()

# Save aggregated metrics for Streamlit
dashboard_metrics = {
    'total_revenue': float(df_clean['TotalAmount'].sum()),
    'total_transactions': int(df_clean['InvoiceNo'].nunique()),
    'total_customers': int(df_clean['CustomerID'].nunique()),
    'total_products': int(df_clean['Description'].nunique()),
    'monthly_data': monthly_data[['YearMonth_str', 'Revenue', 'Transactions']].to_dict(orient='list'),
    'segment_summary': segment_summary.to_dict(orient='records'),
    'top_5_products': top_5_products.to_dict(orient='records'),
    'recommendation_coverage_pct': float(len(popular_products) / len(product_cust_counts) * 100),
    'at_risk_count': int(rfm_df['Segment'].value_counts().get('At Risk', 0))
}

joblib.dump(dashboard_metrics, 'models/dashboard_metrics.pkl')
print("Saved models and dashboard_metrics.pkl successfully.")
