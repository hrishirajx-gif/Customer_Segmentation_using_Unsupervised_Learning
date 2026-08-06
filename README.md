# Customer Segmentation using Unsupervised Learning

Segments customers of a UK-based online retailer into groups based on their purchasing behavior, using RFM analysis and four different clustering techniques. The goal is to identify which customers are worth prioritizing and which ones need re-engagement.

## Data

Transaction data from an online gift retailer, covering all purchases between 01/12/2010 and 09/12/2011. Many of the customers are wholesalers rather than individual shoppers.

## Approach

1. **RFM feature engineering** — for each customer, compute:
   - **Recency**: days since their last purchase
   - **Frequency**: number of transactions
   - **Monetary**: total amount spent
2. **Cleaning** — drop rows with missing values, remove outliers in Amount, Frequency, and Recency using the IQR method, and standardize the three features.
3. **Clustering** — the same RFM features are run through four different algorithms, and results are compared:
   - **K-Means** (k chosen via the elbow method, settled on 3 clusters)
   - **Hierarchical/Agglomerative clustering** (ward linkage, guided by a dendrogram)
   - **DBSCAN** (density-based, also flags outliers as noise)
   - **Gaussian Mixture Model** (probabilistic, handles overlapping clusters)
4. **Cluster interpretation** — each algorithm's clusters are visualized (Amount vs. Frequency, Recency vs. Frequency, Amount vs. Recency) and labeled by customer type: high-value, low-value, regular/moderate spenders.

## Result

The Gaussian Mixture Model was chosen as the final approach — it gave the most interpretable, actionable segments and handles overlap between customer types better than the hard-boundary methods. It settles on three segments:

- **High-value, recent, frequent buyers** — retain with loyalty perks and early access
- **Low-value, infrequent, older customers** — re-engage with targeted discounts
- **Moderate spenders with average engagement** — nurture toward the high-value segment

`Business_Recommendations.ipynb` builds out the concrete strategy for each segment.

## Files

| File | What it is |
|---|---|
| `Customer_Segmentation.ipynb` | Main analysis — RFM feature engineering, all four clustering models, cluster visualization and interpretation |
| `Business_Recommendations.ipynb` | Strategy recommendations per customer segment |

## Running it

You'll need the `OnlineRetail.csv` dataset in the same directory (not included in this repo). Dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Then open the notebook:

```bash
jupyter notebook Customer_Segmentation.ipynb
```

## Notes

- This is a learning project exploring different clustering techniques on the same dataset, not a production pipeline.
- Cluster labels (0, 1, 2...) aren't consistent across algorithms — cluster 0 in K-Means isn't the same segment as cluster 0 in DBSCAN.
