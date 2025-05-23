import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create sample data similar to your structure
np.random.seed(42)
categories = ['explicit', 'presupposition', 'conditional', 'counterfactual', 'imperative']
dimensions = ['form', 'epistemic_stance', 'evidentiality']

data = []
for cat in categories:
    for dim in dimensions:
        data.append({
            'category': cat,
            'dimension': dim,
            'context_pct': np.random.uniform(20, 80),
            'memory_pct': np.random.uniform(10, 70)
        })

category_summary_df = pd.DataFrame(data)
category_summary_df['context_following_rate'] = (
    category_summary_df['context_pct'] / 
    (category_summary_df['context_pct'] + category_summary_df['memory_pct'])
) * 100

print("Original DataFrame:")
print(category_summary_df.head())

# METHOD 1: Sort by average context_pct across all dimensions
avg_by_category = category_summary_df.groupby('category')['context_pct'].mean().sort_values(ascending=False)
category_order = avg_by_category.index.tolist()

plt.figure(figsize=(12, 8))
sns.barplot(data=category_summary_df, y="category", x="context_pct", hue="dimension", 
            order=category_order)
plt.title("Method 1: Sorted by Average Context % (Highest to Lowest)")
plt.tight_layout()
plt.show()

# METHOD 2: Sort by sum of context_pct across all dimensions
sum_by_category = category_summary_df.groupby('category')['context_pct'].sum().sort_values(ascending=False)
category_order_sum = sum_by_category.index.tolist()

plt.figure(figsize=(12, 8))
sns.barplot(data=category_summary_df, y="category", x="context_pct", hue="dimension", 
            order=category_order_sum)
plt.title("Method 2: Sorted by Sum of Context % (Highest to Lowest)")
plt.tight_layout()
plt.show()

# METHOD 3: Sort by a specific dimension (e.g., 'form')
form_data = category_summary_df[category_summary_df['dimension'] == 'form']
form_order = form_data.sort_values('context_pct', ascending=False)['category'].tolist()

plt.figure(figsize=(12, 8))
sns.barplot(data=category_summary_df, y="category", x="context_pct", hue="dimension", 
            order=form_order)
plt.title("Method 3: Sorted by 'form' Dimension Values (Highest to Lowest)")
plt.tight_layout()
plt.show()

# METHOD 4: Sort by context_following_rate instead
avg_following_rate = category_summary_df.groupby('category')['context_following_rate'].mean().sort_values(ascending=False)
following_rate_order = avg_following_rate.index.tolist()

plt.figure(figsize=(12, 8))
sns.barplot(data=category_summary_df, y="category", x="context_following_rate", hue="dimension", 
            order=following_rate_order)
plt.title("Method 4: Sorted by Context Following Rate (Highest to Lowest)")
plt.tight_layout()
plt.show()

# METHOD 5: Manual sorting of the entire DataFrame
category_summary_df_sorted = category_summary_df.sort_values(['category', 'context_pct'], ascending=[True, False])

plt.figure(figsize=(12, 8))
sns.barplot(data=category_summary_df_sorted, y="category", x="context_pct", hue="dimension")
plt.title("Method 5: Pre-sorted DataFrame")
plt.tight_layout()
plt.show()

print("\nCategory order by average context_pct (highest to lowest):")
for i, (cat, avg_val) in enumerate(avg_by_category.items(), 1):
    print(f"{i}. {cat}: {avg_val:.1f}%") 