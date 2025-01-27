import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

customers_path = "Customers.csv"
products_path = "Products.csv"
transactions_path = "Transactions.csv"

customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
transactions = pd.read_csv(transactions_path)

transactions = transactions.merge(products, on="ProductID").merge(customers, on="CustomerID")

customer_features = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': lambda x: x.nunique(),
    'Category': lambda x: x.mode()[0]
}).reset_index()

customer_features = customer_features.rename(columns={
    'TotalValue': 'TotalSpending',
    'Quantity': 'TotalQuantity',
    'ProductID': 'UniqueProductsBought',
    'Category': 'FavoriteCategory'
})

customer_features = pd.get_dummies(customer_features, columns=['FavoriteCategory'])
scaler = MinMaxScaler()
numerical_cols = ['TotalSpending', 'TotalQuantity', 'UniqueProductsBought']
customer_features[numerical_cols] = scaler.fit_transform(customer_features[numerical_cols])

customer_matrix = customer_features.drop(columns=['CustomerID']).values
similarity_matrix = cosine_similarity(customer_matrix)

lookalikes = {}
customer_ids = customer_features['CustomerID'].values

for idx, customer_id in enumerate(customer_ids):
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_3 = [(customer_ids[i], score) for i, score in similarity_scores if i != idx][:3]
    lookalikes[customer_id] = top_3

lookalike_data = []
for customer_id in customer_ids[:20]:
    lookalike_entry = {
        "cust_id": customer_id,
        "lookalikes": [
            {"similar_cust_id": sim[0], "score": round(sim[1], 4)} for sim in lookalikes[customer_id]
        ]
    }
    lookalike_data.append(lookalike_entry)

lookalike_map = {
    row["cust_id"]: row["lookalikes"] for row in lookalike_data
}

lookalike_df = pd.DataFrame({
    "cust_id": list(lookalike_map.keys()),
    "lookalikes": [str(lookalike_map[cust_id]) for cust_id in lookalike_map.keys()]
})
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike recommendations saved to Lookalike.csv:")
print(lookalike_df.head(20))
