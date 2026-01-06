import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.data_preprocessing import preprocess_data
from src.clustering import find_optimal_clusters, train_kmeans
from src.visualization import (
    add_cluster_labels,
    cluster_summary,
    plot_cluster_distribution,
    plot_feature_by_cluster
)

# -----------------------------------
# App Configuration
# -----------------------------------
st.set_page_config(page_title="User Profiling & Segmentation", layout="wide")

st.title("📊 User Profiling & Segmentation System")

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Data Exploration", "User Segmentation", "Business Insights"]
)

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/user_profiles_for_ads.csv")

df = load_data()

# -----------------------------------
# Page 1: Project Overview
# -----------------------------------
if menu == "Project Overview":
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    To develop a user profiling and segmentation system using machine learning
    that helps businesses design more effective ad campaigns.

    **Approach:**  
    - Data preprocessing  
    - K-Means clustering  
    - Cluster interpretation  
    - Interactive visualization using Streamlit
    """)

    st.subheader("Dataset Summary")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.dataframe(df.head())

# -----------------------------------
# Page 2: Data Exploration
# -----------------------------------
elif menu == "Data Exploration":
    st.header("🔍 Data Exploration")

    numeric_cols = df.select_dtypes(exclude="object").columns.tolist()
    feature = st.selectbox("Select a numerical feature", numeric_cols)

    plt.figure()
    df[feature].hist(bins=30)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")
    st.pyplot(plt)

# -----------------------------------
# Page 3: User Segmentation
# -----------------------------------
elif menu == "User Segmentation":
    st.header("🧠 User Segmentation")

    # Preprocess data
    X, preprocessor = preprocess_data(df)

    st.subheader("Find Optimal Number of Clusters")
    K_range, inertia, silhouette_scores = find_optimal_clusters(X)

    col1, col2 = st.columns(2)

    with col1:
        plt.figure()
        plt.plot(K_range, inertia, marker="o")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        st.pyplot(plt)

    with col2:
        plt.figure()
        plt.plot(K_range, silhouette_scores, marker="o")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        st.pyplot(plt)

    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

    # Train model
    model, labels = train_kmeans(X, k)

    df_clustered = add_cluster_labels(df, labels)

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary(df_clustered))

    st.subheader("Cluster Distribution")
    st.pyplot(plot_cluster_distribution(df_clustered))

# -----------------------------------
# Page 4: Business Insights
# -----------------------------------
elif menu == "Business Insights":
    st.header("💡 Business Insights")

    X, _ = preprocess_data(df)
    model, labels = train_kmeans(X, 3)
    df_clustered = add_cluster_labels(df, labels)

    st.subheader("Cluster-wise Insights")
    st.dataframe(cluster_summary(df_clustered))

    feature = st.selectbox(
        "Compare clusters by feature",
        df.select_dtypes(exclude="object").columns.tolist()
    )

    st.pyplot(plot_feature_by_cluster(df_clustered, feature))

    st.markdown("""
    ### 📈 How Businesses Can Use This:
    - Identify high-conversion user segments
    - Optimize ad targeting
    - Personalize marketing strategies
    """)

# -----------------------------------
# Footer
# -----------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Developed by Akanksha Verma**")
