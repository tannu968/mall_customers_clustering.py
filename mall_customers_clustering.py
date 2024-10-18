import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

# Load the dataset
df = pd.read_csv("E:/Mall_customers.csv")

# Display the first and last five rows
print(df.head())
print(df.tail())

# Encode categorical features
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Here we assume 'Age' is categorical for encoding, 
# but it's usually a continuous variable. Consider using normalization/scaling instead.
df['Age'] = le.fit_transform(df['Age'])
df['Annual Income (k$)'] = le.fit_transform(df['Annual Income (k$)'])
df['Spending Score (1-100)'] = le.fit_transform(df['Spending Score (1-100)'])

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Extract relevant features for clustering
df1 = df.iloc[:, [3, 4]].values

# Visualize the data
plt.scatter(df1[:, 0], df1[:, 1], s=10, c="black")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Data Visualization')
plt.show()

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=5, min_samples=5)
labels = dbscan.fit_predict(df)

# Unique clusters
print(f"Unique clusters: {np.unique(labels)}")

# Visualizing the clusters
plt.scatter(df1[labels == -1, 0], df1[labels == -1, 1], s=10, c='black', label='Noise')
for label in np.unique(labels):
    if label != -1:  # Ignore noise
        plt.scatter(df1[labels == label, 0], df1[labels == label, 1], s=10, label=f'Cluster {label}')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('DBSCAN Clustering Results')
plt.legend()
plt.show()
