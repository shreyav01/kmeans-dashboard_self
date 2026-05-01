import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

# ── original function (unchanged) ──────────────────────────────────────────
def kmeans_clustering(n_clusters=3):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, iris.target_names

# ── original plot functions (unchanged) ────────────────────────────────────
def plot_2d_scatter(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['sepal length (cm)'],
                    y=df['sepal width (cm)'],
                    hue=df['Cluster'], palette='viridis')
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("K-Means Clustering (2D View)")
    plt.savefig("static/plot_2d.png")
    plt.close()

def plot_3d_scatter(df):
    fig = px.scatter_3d(df,
                        x='sepal length (cm)',
                        y='sepal width (cm)',
                        z='petal length (cm)',
                        color=df['Cluster'].astype(str),
                        title="K-Means Clustering (3D View)")
    fig.write_html("static/plot_3d.html")

# ── NEW: elbow + silhouette chart ──────────────────────────────────────────
def compute_elbow_data(max_k=10):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(round(silhouette_score(df_scaled, labels), 4))

    return list(k_range), inertias, silhouettes

def plot_elbow_chart():
    k_range, inertias, silhouettes = compute_elbow_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    axes[0].plot(k_range, inertias, 'o-', color='#6366f1', linewidth=2.5, markersize=7)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].set_title('Elbow Method')
    axes[0].axvline(x=3, color='#f59e0b', linestyle='--', alpha=0.7, label='K=3 (chosen)')
    axes[0].legend(facecolor='#1e293b', labelcolor='#94a3b8')

    axes[1].plot(k_range, silhouettes, 's-', color='#10b981', linewidth=2.5, markersize=7)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score per K')
    axes[1].axvline(x=3, color='#f59e0b', linestyle='--', alpha=0.7, label='K=3 (chosen)')
    axes[1].legend(facecolor='#1e293b', labelcolor='#94a3b8')

    plt.tight_layout()
    plt.savefig("static/plot_elbow.png", facecolor='#0f172a')
    plt.close()

# ── NEW: cluster stats summary ─────────────────────────────────────────────
def get_cluster_stats(df):
    features = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']
    stats = []
    for cluster_id in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == cluster_id]
        row = {
            'cluster': int(cluster_id),
            'count': int(len(sub)),
            'pct': round(len(sub) / len(df) * 100, 1),
        }
        for f in features:
            row[f + '_mean'] = round(sub[f].mean(), 2)
        stats.append(row)
    return stats

# ── NEW: silhouette score for current clustering ───────────────────────────
def get_silhouette(n_clusters=3):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(df_scaled)
    return round(silhouette_score(df_scaled, labels), 4)
