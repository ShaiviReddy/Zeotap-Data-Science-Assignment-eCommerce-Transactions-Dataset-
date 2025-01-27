import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

customers_path = "Customers.csv"
products_path = "Products.csv"
transactions_path = "Transactions.csv"

customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
transactions = pd.read_csv(transactions_path)

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

category_revenue = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
top_customers = merged_data.groupby('CustomerID')['TotalValue'].sum().nlargest(5)
merged_data['Month'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_trend = merged_data.groupby('Month')['TransactionID'].count()
top_products = merged_data.groupby('ProductName')['Quantity'].sum().nlargest(5)
region_count = customers['Region'].value_counts()

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Key Insights from EDA", fontsize=16)

sns.barplot(x=region_count.index, y=region_count.values, ax=axes[0, 0], palette="Blues_d")
axes[0, 0].set_title("Customer Distribution by Region")
axes[0, 0].set_xlabel("Region")
axes[0, 0].set_ylabel("Number of Customers")

category_revenue.plot(kind='bar', color='skyblue', ax=axes[0, 1])
axes[0, 1].set_title("Revenue by Product Category")
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("Total Revenue")

monthly_trend.plot(ax=axes[1, 0], marker='o', color='green')
axes[1, 0].set_title("Monthly Transactions Trend")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Number of Transactions")

top_customers.plot(kind='bar', color='orange', ax=axes[1, 1])
axes[1, 1].set_title("Top 5 Customers by Revenue")
axes[1, 1].set_xlabel("CustomerID")
axes[1, 1].set_ylabel("Total Revenue")

top_products.plot(kind='bar', color='purple', ax=axes[2, 0])
axes[2, 0].set_title("Top 5 Products by Quantity Sold")
axes[2, 0].set_xlabel("Product Name")
axes[2, 0].set_ylabel("Quantity Sold")

axes[2, 1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("eda_summary.png")
plt.show()
