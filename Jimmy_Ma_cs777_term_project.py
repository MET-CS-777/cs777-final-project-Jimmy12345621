
# Student: Jimmy Ma
# Project: Clustering Analysis of Utility Energy Consumption by ZIP Code
# Dataset: Utility Energy Registry – Monthly ZIP Code Energy Usage
# Source: https://data.ny.gov/Energy-Environment/Utility-Energy-Registry-Monthly-ZIP-Code-Energy-Us/tzb9-c2c6/data_preview

import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator


def parse_georeference(point_str):

    if point_str is None:
        return None, None
    coords = re.findall(r"[-\d\.]+", point_str)
    if len(coords) == 2:
        lon, lat = coords  # 'POINT (lon lat)'
        return float(lat), float(lon)
    return None, None


def find_optimal_k(ks, costs):
    """
    Determine the optimal number of clusters using the maximum distance method.
    """
    k1, c1 = ks[0], costs[0]
    k2, c2 = ks[-1], costs[-1]
    vec_norm = math.sqrt((k2 - k1) ** 2 + (c2 - c1) ** 2)

    max_distance = -1
    optimal_k = ks[0]
    for k, c in zip(ks, costs):
        distance = abs((c2 - c1) * k - (k2 - k1) * c + k2 * c1 - c2 * k1) / vec_norm
        if distance > max_distance:
            max_distance = distance
            optimal_k = k
    return optimal_k


def main():
    # Initialize SparkSession
    spark = SparkSession.builder.appName("EnergyClusterAnalysis").getOrCreate()

    # Adjust file path as needed
    here = os.path.abspath('Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv')
    input_dir = os.path.abspath(os.path.join(here, os.pardir))
    data_path = os.path.join(input_dir, "Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv")
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_path)

    print("Data Schema:")
    df.printSchema()

    # Data cleaning: also ensure 'Georeference' is not null
    df = df.dropna(subset=["zip_code", "value", "Georeference"])
    df = df.withColumn("value", col("value").cast("double"))

    # Feature assembly
    assembler = VectorAssembler(inputCols=["value"], outputCol="features")
    feature_df = assembler.transform(df)

    # Split data (70% training, 30% testing)
    training_data, testing_data = feature_df.randomSplit([0.7, 0.3], seed=42)

    # --- K-Means: Elbow Method ---
    print("Elbow Method: Evaluating cost for different k values on training data")
    max_k = 10
    ks = list(range(2, max_k + 1))
    costs = []

    for k in ks:
        kmeans_temp = KMeans(featuresCol="features", predictionCol="prediction", k=k, seed=42)
        model_temp = kmeans_temp.fit(training_data)
        cost = model_temp.summary.trainingCost
        costs.append(cost)
        print(f"k = {k} -> Cost (WSSSE): {cost}")

    plt.figure(figsize=(8, 6))
    plt.plot(ks, costs, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Cost (WSSSE)')
    plt.title('Elbow Method For Optimal k (Training Data)')
    plt.grid(True)
    plt.savefig("elbow_plot.png")
    plt.show()

    optimal_k = find_optimal_k(ks, costs)
    print("Automatically determined optimal k is:", optimal_k)

    # --- K-Means on Testing Data ---
    kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=optimal_k, seed=42)
    kmeans_model = kmeans.fit(training_data)
    kmeans_predictions = kmeans_model.transform(testing_data)

    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
    kmeans_silhouette = evaluator.evaluate(kmeans_predictions)
    print("K-Means Silhouette Score on Testing Data = {:.4f}".format(kmeans_silhouette))

    kmeans_centers = kmeans_model.clusterCenters()
    print("K-Means Cluster Centers:")
    for center in kmeans_centers:
        print(center)

    # Convert predictions to Pandas (include 'Georeference')
    pdf_kmeans = kmeans_predictions.select("zip_code", "value", "prediction", "Georeference").toPandas()
    pdf_kmeans['jitter'] = np.random.rand(len(pdf_kmeans))

    # Sort clusters by center value
    kmeans_center_values = [c[0] for c in kmeans_centers]
    sorted_indices_kmeans = sorted(range(len(kmeans_center_values)), key=lambda i: kmeans_center_values[i])

    # Define cluster labels
    cluster_labels_list = [
        "Low-Consumption Cluster",
        "Moderate-Consumption Cluster",
        "High-Consumption Cluster",
        "Very-High-Consumption Cluster"
    ]

    cluster_labels_kmeans = {}
    for rank, cluster_idx in enumerate(sorted_indices_kmeans):
        cluster_labels_kmeans[cluster_idx] = cluster_labels_list[rank]

    # Map predictions to labels
    pdf_kmeans['cluster_label'] = pdf_kmeans['prediction'].map(cluster_labels_kmeans)

    # --- K-Means Scatter Plot ---
    plt.figure(figsize=(10, 6))
    unique_clusters = sorted(pdf_kmeans['prediction'].unique())
    for cluster_val in unique_clusters:
        subset = pdf_kmeans[pdf_kmeans['prediction'] == cluster_val]
        plt.scatter(
            subset['value'],
            subset['jitter'],
            label=cluster_labels_kmeans[cluster_val],
            alpha=0.7
        )
    for center in kmeans_centers:
        plt.axvline(x=center[0], color='black', linestyle='--')

    plt.xlabel('Energy Usage (value)')
    plt.ylabel('Jitter (for visualization)')
    plt.title('K-Means: Scatter Plot of Energy Usage by Cluster (Testing Data)')
    plt.legend()
    plt.savefig("kmeans_cluster_scatter.png")
    plt.show()

    # --- K-Means Heatmap ---
    pdf_kmeans['energy_bin'] = pd.cut(pdf_kmeans['value'], bins=10)
    heatmap_data_kmeans = pdf_kmeans.groupby(['prediction', 'energy_bin'], observed=False).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data_kmeans, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("K-Means: Heatmap of Energy Usage Bins per Cluster (Testing Data)")
    plt.xlabel("Energy Usage Bins")
    plt.ylabel("Cluster")
    plt.savefig("kmeans_cluster_heatmap.png")
    plt.show()

    # --- K-Means Folium Map ---
    print("Generating K-Means cluster map...")
    m_kmeans = folium.Map(location=[40.7128, -74.0060], zoom_start=7)

    color_map = {
        "Low-Consumption Cluster": "green",
        "Moderate-Consumption Cluster": "blue",
        "High-Consumption Cluster": "orange",
        "Very-High-Consumption Cluster": "red"
    }

    for idx, row in pdf_kmeans.iterrows():
        lat, lon = parse_georeference(row["Georeference"])
        if lat is not None and lon is not None:
            cluster_label = row["cluster_label"]
            color = color_map.get(cluster_label, "gray")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"ZIP: {row['zip_code']}\n"
                    f"Usage: {row['value']}\n"
                    f"Cluster: {cluster_label}"
                )
            ).add_to(m_kmeans)

    m_kmeans.save("kmeans_cluster_map.html")
    print("Map saved as kmeans_cluster_map.html")

    # --- GMM Clustering ---
    gmm = GaussianMixture(featuresCol="features", predictionCol="prediction", k=optimal_k, seed=42)
    gmm_model = gmm.fit(training_data)
    gmm_predictions = gmm_model.transform(testing_data)

    gmm_silhouette = evaluator.evaluate(gmm_predictions)
    print("GMM Silhouette Score on Testing Data = {:.4f}".format(gmm_silhouette))

    gmm_centers = [comp.mean for comp in gmm_model.gaussians]
    print("GMM Cluster Centers (means):")
    for center in gmm_centers:
        print(center)

    pdf_gmm = gmm_predictions.select("zip_code", "value", "prediction", "Georeference").toPandas()
    pdf_gmm['jitter'] = np.random.rand(len(pdf_gmm))

    gmm_center_values = [c[0] for c in gmm_centers]
    sorted_indices_gmm = sorted(range(len(gmm_center_values)), key=lambda i: gmm_center_values[i])

    cluster_labels_gmm_list = [
        "Low-Consumption Cluster",
        "Moderate-Consumption Cluster",
        "High-Consumption Cluster",
        "Very-High-Consumption Cluster"
    ]

    cluster_labels_gmm = {}
    for rank, cluster_idx in enumerate(sorted_indices_gmm):
        cluster_labels_gmm[cluster_idx] = cluster_labels_gmm_list[rank]

    pdf_gmm['cluster_label'] = pdf_gmm['prediction'].map(cluster_labels_gmm)

    # --- GMM Scatter Plot ---
    plt.figure(figsize=(10, 6))
    unique_clusters = sorted(pdf_gmm['prediction'].unique())
    for cluster_val in unique_clusters:
        subset = pdf_gmm[pdf_gmm['prediction'] == cluster_val]
        plt.scatter(
            subset['value'],
            subset['jitter'],
            label=cluster_labels_gmm[cluster_val],
            alpha=0.7
        )
    for center in gmm_centers:
        plt.axvline(x=center[0], color='black', linestyle='--')

    plt.xlabel('Energy Usage (value)')
    plt.ylabel('Jitter (for visualization)')
    plt.title('GMM: Scatter Plot of Energy Usage by Cluster (Testing Data)')
    plt.legend()
    plt.savefig("gmm_cluster_scatter.png")
    plt.show()

    # --- GMM Heatmap ---
    pdf_gmm['energy_bin'] = pd.cut(pdf_gmm['value'], bins=10)
    heatmap_data_gmm = pdf_gmm.groupby(['prediction', 'energy_bin'], observed=False).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data_gmm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("GMM: Heatmap of Energy Usage Bins per Cluster (Testing Data)")
    plt.xlabel("Energy Usage Bins")
    plt.ylabel("Cluster")
    plt.savefig("gmm_cluster_heatmap.png")
    plt.show()

    # --- GMM Folium Map ---
    print("Generating GMM cluster map...")
    m_gmm = folium.Map(location=[40.7128, -74.0060], zoom_start=7)

    for idx, row in pdf_gmm.iterrows():
        lat, lon = parse_georeference(row["Georeference"])
        if lat is not None and lon is not None:
            cluster_label = row["cluster_label"]
            color = color_map.get(cluster_label, "gray")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"ZIP: {row['zip_code']}\n"
                    f"Usage: {row['value']}\n"
                    f"Cluster: {cluster_label}"
                )
            ).add_to(m_gmm)

    m_gmm.save("gmm_cluster_map.html")
    print("Map saved as gmm_cluster_map.html")

    # --- Save Predictions to CSV ---
    kmeans_predictions.select("zip_code", "value", "prediction") \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv("kmeans_energy_cluster_output.csv")

    gmm_predictions.select("zip_code", "value", "prediction") \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv("gmm_energy_cluster_output.csv")

    spark.stop()


if __name__ == "__main__":
    main()
