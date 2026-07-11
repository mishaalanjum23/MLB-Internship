from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X = scaler.fit_transform(X) 

# Applying PCA
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X) 

# Visualize transformed data
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c = kmeans.labels_)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters after PCA")
plt.show()