# energy_cluster.py
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
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator


def find_optimal_k(ks, costs):
    """
    Determine the optimal number of clusters using the maximum distance method.

    Parameters:
        ks (list of int): List of k values.
        costs (list of float): Corresponding cost (WSSSE) for each k.

    Returns:
        optimal_k (int): The optimal k value.
    """
    # Define first and last points on the curve
    k1, c1 = ks[0], costs[0]
    k2, c2 = ks[-1], costs[-1]
    # Compute the norm of the line from first to last
    vec_norm = math.sqrt((k2 - k1) ** 2 + (c2 - c1) ** 2)

    max_distance = -1
    optimal_k = ks[0]
    # For each point, compute the perpendicular distance to the line connecting the first and last points.
    for k, c in zip(ks, costs):
        # Distance formula from point (k, c) to line through (k1, c1) and (k2, c2)
        distance = abs((c2 - c1) * k - (k2 - k1) * c + k2 * c1 - c2 * k1) / vec_norm
        if distance > max_distance:
            max_distance = distance
            optimal_k = k
    return optimal_k
def main():
    # Initialize SparkSession
    spark = SparkSession.builder.appName("EnergyClusterAnalysis").getOrCreate()
    here = os.path.abspath('Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv')
    input_dir = os.path.abspath(os.path.join(here, os.pardir))
    data_path = os.path.join(input_dir, "Utility_Energy_Registry_Monthly_ZIP_Code_Energy_Use__2016-2021_20250220.csv")
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_path)
    print("Data Schema:")
    df.printSchema()
    # Data cleaning: drop rows with missing key fields (using the correct column names)
    df = df.dropna(subset=["zip_code", "value"])
    df = df.withColumn("value", col("value").cast("double"))

    assembler = VectorAssembler(inputCols=["value"], outputCol="features")
    feature_df = assembler.transform(df)

    # Split data into training (70%) and testing (30%) subsets
    training_data, testing_data = feature_df.randomSplit([0.7, 0.3], seed=42)

    # ----- K-Means: Determine optimal k using the elbow method on training data -----
    print("Elbow Method: Evaluating cost for different k values on training data")
    max_k = 10  # Evaluate k values from 2 to 7
    ks = list(range(2, max_k + 1))
    costs = []

    for k in ks:
        kmeans_temp = KMeans(featuresCol="features", predictionCol="prediction", k=k, seed=42)
        model_temp = kmeans_temp.fit(training_data)
        cost = model_temp.summary.trainingCost  # WSSSE for the given k
        costs.append(cost)
        print("k =", k, "-> Cost (WSSSE):", cost)

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

    # --- K-Means Clustering on Testing Data ---
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

    # Convert predictions to Pandas
    pdf_kmeans = kmeans_predictions.select("zip_code", "value", "prediction").toPandas()
    pdf_kmeans['jitter'] = np.random.rand(len(pdf_kmeans))

    # 1) Sort clusters by their center value
    kmeans_center_values = [c[0] for c in kmeans_centers]  # Each center is a 1D array [value]
    sorted_indices_kmeans = sorted(range(len(kmeans_center_values)), key=lambda i: kmeans_center_values[i])

    # 2) Define labels for each cluster (assuming 4 clusters)
    cluster_labels_list = [
        "Low-Consumption Cluster",
        "Moderate-Consumption Cluster",
        "High-Consumption Cluster",
        "Very-High-Consumption Cluster"
    ]

    # 3) Create a dictionary mapping cluster index -> label
    cluster_labels_kmeans = {}
    for rank, cluster_idx in enumerate(sorted_indices_kmeans):
        cluster_labels_kmeans[cluster_idx] = cluster_labels_list[rank]

    # 4) Map predictions to labels
    pdf_kmeans['cluster_label'] = pdf_kmeans['prediction'].map(cluster_labels_kmeans)

    # 5) Scatter plot, grouping by cluster_label
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

    # Draw vertical lines at cluster centers
    for center in kmeans_centers:
        plt.axvline(x=center[0], color='red', linestyle='--')

    plt.xlabel('Energy Usage (value)')
    plt.ylabel('Jitter (for visualization)')
    plt.title('K-Means: Scatter Plot of Energy Usage by Cluster (Testing Data)')
    plt.legend()
    plt.savefig("kmeans_cluster_scatter.png")
    plt.show()

    # Heatmap for K-Means
    pdf_kmeans['energy_bin'] = pd.cut(pdf_kmeans['value'], bins=10)
    heatmap_data_kmeans = pdf_kmeans.groupby(['prediction', 'energy_bin'], observed=False).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data_kmeans, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("K-Means: Heatmap of Energy Usage Bins per Cluster (Testing Data)")
    plt.xlabel("Energy Usage Bins")
    plt.ylabel("Cluster")
    plt.savefig("kmeans_cluster_heatmap.png")
    plt.show()

    # --- GMM Clustering ---
    gmm = GaussianMixture(featuresCol="features", predictionCol="prediction", k=optimal_k, seed=42)
    gmm_model = gmm.fit(training_data)
    gmm_predictions = gmm_model.transform(testing_data)

    gmm_silhouette = evaluator.evaluate(gmm_predictions)
    print("GMM Silhouette Score on Testing Data = {:.4f}".format(gmm_silhouette))

    # Retrieve GMM cluster centers (the means)
    gmm_centers = [comp.mean for comp in gmm_model.gaussians]
    print("GMM Cluster Centers (means):")
    for center in gmm_centers:
        print(center)

    # Convert predictions to Pandas
    pdf_gmm = gmm_predictions.select("zip_code", "value", "prediction").toPandas()
    pdf_gmm['jitter'] = np.random.rand(len(pdf_gmm))

    # 1) Sort clusters by their center value
    gmm_center_values = [c[0] for c in gmm_centers]
    sorted_indices_gmm = sorted(range(len(gmm_center_values)), key=lambda i: gmm_center_values[i])

    # 2) Define labels (same as K-Means, assuming 4 clusters)
    cluster_labels_list = [
        "Low-Consumption Cluster",
        "Moderate-Consumption Cluster",
        "High-Consumption Cluster",
        "Very-High-Consumption Cluster"
    ]

    # 3) Create a dictionary mapping cluster index -> label
    cluster_labels_gmm = {}
    for rank, cluster_idx in enumerate(sorted_indices_gmm):
        cluster_labels_gmm[cluster_idx] = cluster_labels_list[rank]

    # 4) Map predictions to labels
    pdf_gmm['cluster_label'] = pdf_gmm['prediction'].map(cluster_labels_gmm)

    # 5) Scatter plot, grouping by cluster_label
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

    # Draw vertical lines at cluster centers
    for center in gmm_centers:
        plt.axvline(x=center[0], color='red', linestyle='--')

    plt.xlabel('Energy Usage (value)')
    plt.ylabel('Jitter (for visualization)')
    plt.title('GMM: Scatter Plot of Energy Usage by Cluster (Testing Data)')
    plt.legend()
    plt.savefig("gmm_cluster_scatter.png")
    plt.show()

    # Heatmap for GMM
    pdf_gmm['energy_bin'] = pd.cut(pdf_gmm['value'], bins=10)
    heatmap_data_gmm = pdf_gmm.groupby(['prediction', 'energy_bin'], observed=False).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data_gmm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("GMM: Heatmap of Energy Usage Bins per Cluster (Testing Data)")
    plt.xlabel("Energy Usage Bins")
    plt.ylabel("Cluster")
    plt.savefig("gmm_cluster_heatmap.png")
    plt.show()

    # Save predictions to a single CSV file (using coalesce)
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