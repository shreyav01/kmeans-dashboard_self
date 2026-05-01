from flask import Flask, render_template, jsonify, request
from kmeans_irss import (
    kmeans_clustering, plot_2d_scatter, plot_3d_scatter,
    plot_elbow_chart, get_cluster_stats, get_silhouette
)

app = Flask(__name__)

@app.route('/')
def home():
    # original logic preserved — default K=3
    df, target_names = kmeans_clustering(n_clusters=3)
    plot_2d_scatter(df)
    plot_3d_scatter(df)
    plot_elbow_chart()

    cluster_stats = get_cluster_stats(df)
    silhouette = get_silhouette(n_clusters=3)

    return render_template("index.html",
                           cluster_stats=cluster_stats,
                           silhouette=silhouette,
                           current_k=3)

# NEW: AJAX endpoint — re-runs clustering with chosen K
@app.route('/recluster')
def recluster():
    k = int(request.args.get('k', 3))
    k = max(2, min(k, 8))  # clamp between 2 and 8

    df, _ = kmeans_clustering(n_clusters=k)
    plot_2d_scatter(df)
    plot_3d_scatter(df)

    cluster_stats = get_cluster_stats(df)
    silhouette = get_silhouette(n_clusters=k)

    return jsonify({
        'cluster_stats': cluster_stats,
        'silhouette': silhouette,
        'k': k
    })

if __name__ == '__main__':
    app.run(debug=True)
