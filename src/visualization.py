import pandas as pd
import matplotlib.pyplot as plt


def add_cluster_labels(df, labels):
    """
    Add cluster labels to original dataframe
    """
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    return df_clustered


def cluster_summary(df_clustered):
    """
    Generate cluster-wise summary statistics
    """
    summary = df_clustered.groupby("Cluster").mean(numeric_only=True)
    return summary


def plot_cluster_distribution(df_clustered):
    """
    Plot number of users per cluster
    """
    cluster_counts = df_clustered["Cluster"].value_counts().sort_index()

    plt.figure()
    cluster_counts.plot(kind="bar")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.title("User Distribution Across Clusters")
    plt.tight_layout()

    return plt


def plot_feature_by_cluster(df_clustered, feature):
    """
    Plot average feature value per cluster
    """
    cluster_feature_mean = df_clustered.groupby("Cluster")[feature].mean()

    plt.figure()
    cluster_feature_mean.plot(kind="bar")
    plt.xlabel("Cluster")
    plt.ylabel(f"Average {feature}")
    plt.title(f"{feature} by Cluster")
    plt.tight_layout()

    return plt
