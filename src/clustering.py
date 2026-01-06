import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


def find_optimal_clusters(X, max_k=10):
    """
    Uses Elbow Method and Silhouette Score to find optimal K
    """
    inertia = []
    silhouette_scores = []

    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    return K_range, inertia, silhouette_scores


def train_kmeans(X, n_clusters):
    """
    Train final KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    return kmeans, labels


def save_model(model, filepath="model/kmeans_model.pkl"):
    """
    Save trained model
    """
    joblib.dump(model, filepath)
