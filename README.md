This project implements KMeans clustering using Python and the scikit-learn library to group students based on their grades in three subjects: Math, Science, and English. The dataset consists of randomly generated grades, and the goal is to cluster students into 3 distinct categories based on their performance.

Data Generation:
A sample dataset of student grades is generated using numpy, with random scores between 50 and 100 for each subject (Math, Science, and English).

Data Standardization:
The grades are standardized using StandardScaler from scikit-learn to ensure each subject contributes equally to the clustering. This step is crucial when features have different ranges, as it avoids one feature (e.g., higher grades) dominating the clustering process.

KMeans Clustering:
The KMeans algorithm is applied to group students into 3 clusters based on their standardized grades.
After clustering, the predicted cluster labels are added to the original dataset.

Cluster Centroids:
The centroids (representing the "average" student in each cluster) are computed and displayed. These centroids help in understanding the general performance of students in each group.
The centroids are also inverse-transformed to bring them back to the original scale of grades for better interpretation.

Visualization:
A 2D scatter plot is generated to visualize how the students are grouped into clusters based on Math and Science grades.
The centroids are plotted to show the center of each cluster, making it easier to interpret the groupings visually.
