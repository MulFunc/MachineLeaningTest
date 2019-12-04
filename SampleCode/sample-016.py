# Data loading
from sklearn.datasets import load_wine
data = load_wine()
X = data.data[:, [0, 9]]

# Learning & Prediction
from sklearn.cluster import KMeans
cluster_num = 3
model = KMeans(n_clusters = cluster_num)
pred = model.fit_predict(X)
