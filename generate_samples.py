import numpy as np
import tensorflow as tf
from functions import create_samples, choose_random_centroids, assign_to_nearest, update_centroids
from functions import plot_clusters

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters, seed)

nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)
updated_centroid_values = []
for i in range(10):
    nearest_indices = assign_to_nearest(samples, updated_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

    model = tf.initialize_all_variables()
    with tf.Session() as session:
        sample_values = session.run(samples)
        updated_centroid_value = session.run(updated_centroids)
        print(updated_centroid_value)
        updated_centroid_values.append(updated_centroid_value)
        print(updated_centroid_values)

plot_clusters(sample_values, updated_centroid_values, n_samples_per_cluster)
