import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("C:/Users/Tanvi/dataset/grocery transactions.xlsx", header=None, names=['Date', 'Items'])
print("T106 Tanvi Sakhale\n", "~~" * 50, "\nFirst 3 Rows:\n", df.head(3), f"\nShape: {df.shape}\n", "~~" * 50)

# One-hot encoding
data = [x.split(',') for x in df['Items']]
encoder = TransactionEncoder()
df_encoded = pd.DataFrame(encoder.fit_transform(data), columns=encoder.columns_)

# Frequent itemsets and association rules
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True).sort_values('support', ascending=False)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2).sort_values('confidence', ascending=False)

# Print rules and top 10 items
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], "\n", "~~" * 50)
print("Top 10 Items by Support:\n", frequent_itemsets.nlargest(10, 'support'))

# Visualizations
plt.figure(figsize=(8, 5))
sns.barplot(x='support', y='itemsets', data=frequent_itemsets.nlargest(10, 'support'), palette='viridis', hue='itemsets', legend=False)
plt.title('Top 10 Items by Support'); plt.show()

rules['pair'] = rules['antecedents'].apply(lambda x: ', '.join(x)) + " -> " + rules['consequents'].apply(lambda x: ', '.join(x))
plt.figure(figsize=(8, 5))
sns.lineplot(data=rules, x='pair', y='confidence', marker='o', color='blue')
plt.xticks(rotation=45, ha='right'); plt.title('Confidence of Item Pairs'); plt.show()

top_lift = rules.nlargest(10, 'lift')
top_lift['pair'] = top_lift['antecedents'].apply(lambda x: ', '.join(x)) + " -> " + top_lift['consequents'].apply(lambda x: ', '.join(x))
plt.figure(figsize=(8, 5))
sns.barplot(x='lift', y='pair', data=top_lift, palette='plasma', hue='pair', legend=False)
plt.title('Top 10 Item Pairs by Lift'); plt.show()

# Added Graph: Top 10 Frequent Itemsets by Support
plt.figure(figsize=(8, 5))
sns.barplot(x='support', y='itemsets', data=frequent_itemsets.head(10), palette='magma', hue='itemsets', legend=False)
plt.title('Top 10 Frequent Itemsets by Support'); plt.show()
