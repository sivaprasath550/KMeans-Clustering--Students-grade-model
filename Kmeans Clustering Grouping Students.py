import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  
num_students = 100   
grades = {
    'Math': np.random.randint(50, 100, num_students),     
    'Science': np.random.randint(50, 100, num_students), 
    'English': np.random.randint(50, 100, num_students)   
}

students_df = pd.DataFrame(grades)  

scaler = StandardScaler()
students_scaled = scaler.fit_transform(students_df)

num_clusters = 3  

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
students_df['Cluster'] = kmeans.fit_predict(students_scaled) 

print("Cluster Centers (Standardized):")
print(kmeans.cluster_centers_)

cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (Original Scale):")
print(cluster_centers_original)

plt.figure(figsize=(10, 6))
plt.scatter(students_df['Math'], students_df['Science'], c=students_df['Cluster'], cmap='viridis', marker='o', edgecolor='k')

plt.scatter(cluster_centers_original[:, 0], cluster_centers_original[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('KMeans Clustering of Students Based on Grades (3 Clusters with Standardization)')
plt.xlabel('Math Grades')
plt.ylabel('Science Grades')
plt.legend()
plt.grid()
plt.show()
