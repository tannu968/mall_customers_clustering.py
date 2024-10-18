# Mall Customers Clustering 

## Overview
This project analyzes customer data from a mall dataset to identify distinct customer segments using the DBSCAN clustering algorithm. The focus is on clustering customers based on their annual income and spending scores.

## Dataset
The dataset `Mall_customers.csv` contains information about customers, including:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

## Steps
1. Load the dataset and explore its structure.
2. Preprocess the data by encoding categorical features.
3. Visualize the data points based on annual income and spending score.
4. Apply the DBSCAN clustering algorithm to group customers.
5. Visualize the resulting clusters.

## Usage
To run this project, make sure you have Python installed along with the required libraries. Execute the script:

```bash
python mall_customers_clustering.py
# mall_customers_clustering.py

Results
The output will display a scatter plot showing the clusters formed by DBSCAN, with noise points represented separately.
