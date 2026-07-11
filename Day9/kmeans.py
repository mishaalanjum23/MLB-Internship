import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load and explore dataset using pandas
iris = load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

print(df.head())
print(df.describe())
df.info()

X = iris.data

# choose value of k(elbow method)
inertialist = []
for k in range (1, 11):
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(X)
    inertialist.append(model.inertia_)

plt.plot(range(1,11), inertialist, marker = "o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(X)

# Visualize the clusters using a scatter plot.
plt.scatter(X[:,0], X[:,1], c = kmeans.labels_)
plt.xlabel("Sepal Length(cm)")
plt.ylabel("Sepal Width(cm)")
plt.title("K-Means Clustering of Iris Dataset")
plt.show()