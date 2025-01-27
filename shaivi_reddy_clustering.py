import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

customers_path = "Customers.csv"
transactions_path = "Transactions.csv"

customers = pd.read_csv(customers_path)
transactions = pd.read_csv(transactions_path)

transactions_agg = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count'
}).reset_index()
transactions_agg = transactions_agg.rename(columns={
    'TotalValue': 'TotalSpending',
    'TransactionID': 'TransactionCount'
})

customer_data = customers.merge(transactions_agg, on="CustomerID", how="left")
customer_data.fillna(0, inplace=True)

customer_data['SignupDate'] = pd.to_datetime(customer_data['SignupDate'])
customer_data['TenureDays'] = (pd.Timestamp.now() - customer_data['SignupDate']).dt.days

customer_data = pd.get_dummies(customer_data, columns=['Region'], drop_first=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[[
    'TotalSpending', 'TransactionCount', 'TenureDays'
]])

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

db_index = davies_bouldin_score(scaled_data, customer_data['Cluster'])
silhouette_avg = silhouette_score(scaled_data, customer_data['Cluster'])

customer_data['DB_Index'] = db_index
customer_data['Silhouette_Score'] = silhouette_avg

customer_data[['CustomerID', 'Cluster', 'DB_Index', 'Silhouette_Score']].to_csv("CustomerClusters_with_metrics.csv", index=False)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_data[:, 0],
    y=scaled_data[:, 1],
    hue=customer_data['Cluster'],
    palette="viridis",
    s=100
)
plt.title(f"Customer Clusters (n_clusters={n_clusters})")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend(title="Cluster")

plt.savefig("customer_clusters_plot.png")

plt.show()

# Print clustering metrics only if they are different from the previous output
if 'db_index_printed' not in globals():
    print(f"Number of clusters: {n_clusters}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    db_index_printed = True
