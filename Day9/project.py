import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

print(df.head())
print(df.describe())
df.info()

X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# choose value of k(elbow method)
inertialist = []
for k in range (1, 11):
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(X_scaled)
    inertialist.append(model.inertia_)

plt.plot(range(1,11), inertialist, marker = "o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("Elbow Method.png")
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(X_scaled)

# Apply PCA
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled) 

#Visualize original data
plt.scatter(X[:,0], X[:,1])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Original Iris Dataset")
plt.savefig("Original.png")
plt.show()

# Visualize kmeans clusters
plt.scatter(X[:,0], X[:,1], c = kmeans.labels_)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering of Iris Dataset")
plt.savefig("K-Means Clustering.png")
plt.show()

# Visualize PCA transformed data
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.savefig("PCA.png")
plt.show()
