import matplotlib.pyplot as plt
from resnet_model.classifier import Classifier
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.cluster import KMeans
import random

def get_feature_vectors(dataset):
    # select support images (first and last <shot> images)
    x, y, ID = dataset[0], dataset[1], dataset[2]
    y = np.array(y)
    y_bool_debris = y == 1
    y_bool_nondebris = y == 0
    debris_idx = np.where(y_bool_debris)[0]
    non_debris_idx = np.where(y_bool_nondebris)[0]
    all_index = np.concatenate((debris_idx, non_debris_idx), axis=None)
    x = torch.stack(x)

    model = Classifier.load_from_checkpoint('resnet_model/checkpoints/resnet18_2022-11-02_14:08:50/epoch=108-val_accuracy=0.95.ckpt')
    model.eval()

    feature_vectors_all = []
    feature_vectors_debris = []
    feature_vectors_nondebris = []
    for i in tqdm(debris_idx):
        model(x[i].unsqueeze(0))
        feature_vector = model.features  # size = [1, 152]
        feature_vector_array = feature_vector[0].detach().numpy()[0]  # size = [1, 152]
        feature_vectors_debris.append(feature_vector_array)
        feature_vectors_all.append(feature_vector_array)
    for i in tqdm(non_debris_idx):
        model(x[i].unsqueeze(0))
        feature_vector_n = model.features  # size = [1, 152]
        feature_vector_array_n = feature_vector_n[0].detach().numpy()[0]  # size = [1, 152]
        feature_vectors_nondebris.append(feature_vector_array_n)
        feature_vectors_all.append(feature_vector_array_n)
    feature_vectors_debris = np.array(feature_vectors_debris)
    feature_vectors_nondebris = np.array(feature_vectors_nondebris)

    feature_vectors_all = np.array(feature_vectors_all) # size = [size_of_dataset, 512]
    y_all = np.concatenate((y[debris_idx], y[non_debris_idx]), axis=None) # size = size_of_dataset

    print("Features extracted.")
    return feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all

def pca_visualization(pca_components, feature_vectors_all, y_all):
    pca = PCA(n_components=pca_components)
    pca.fit(feature_vectors_all)
    vectors_transformed = pca.transform(feature_vectors_all)

    Xax = vectors_transformed[:, 0]
    Yax = vectors_transformed[:, 1]
    Zax = vectors_transformed[:, 2]

    cdict = {0: 'b', 1: 'r'}
    label = {0: 'Non-debris', 1: 'Debris'}

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    for l in np.unique(y_all):
        ix = np.where(y_all == l)
        ax.scatter(Xax[ix],
                   Yax[ix],
                   Zax[ix],
                   c=cdict[l],
                   s=60,
                   label=label[l])

    ax.set_xlabel("PC1",
                  fontsize=12)
    ax.set_ylabel("PC2",
                  fontsize=12)
    ax.set_zlabel("PC3",
                  fontsize=12)

    ax.view_init(30, 125)
    ax.legend()
    plt.title("3D PCA plot")
    plt.show()

def get_good_and_bad_samples(region, number_of_sets):
    with open('datasets/accuracies_and_idx_1-shot_09-01-2023.pickle', 'rb') as data:
        all_sets = pickle.load(data)
    accuracies = all_sets[0][region]
    idx = all_sets[1][region]

    indices, acc_sorted = zip(*sorted(enumerate(accuracies), key=itemgetter(1)))
    best_id = np.array(indices[-number_of_sets:])
    worst_id = np.array(indices[:number_of_sets])
    best_samples = np.concatenate([idx[k] for k in best_id])
    worst_samples = np.concatenate([idx[l] for l in worst_id])

    return best_samples, worst_samples

def pca_good_samples(pca_components, feature_vectors_all, y_all, best_samples):
    pca = PCA(n_components=pca_components)
    pca.fit(feature_vectors_all)
    y_good = y_all[best_samples]
    vectors_good = feature_vectors_all[best_samples]

    Xax = vectors_good[:, 0]
    Yax = vectors_good[:, 1]

    cdict = {0: 'b', 1: 'r'}
    label = {0: 'Non-debris', 1: 'Debris'}

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)

    for l in np.unique(y_good):
        ix = np.where(y_good == l)
        ax.scatter(Xax[ix],
                   Yax[ix],
                   c=cdict[l],
                   s=60,
                   label=label[l])

    ax.set_xlabel("PC1",
                  fontsize=12)
    ax.set_ylabel("PC2",
                  fontsize=12)

    ax.legend()
    plt.title("2D PCA plot for good samples")
    plt.show()

def cluster_all_features(feature_vectors_all, n_clusters, *, visualize: bool = False):
    pca = PCA(n_components=3)
    pca.fit(feature_vectors_all)
    vectors_transformed = pca.transform(feature_vectors_all)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors_transformed)
    labels = kmeans.labels_

    if visualize == True:
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(vectors_transformed[:, 0], vectors_transformed[:, 1], vectors_transformed[:, 2], c=labels.astype(float), s=60)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_zlabel("PC3", fontsize=12)
        ax.set_title(str(n_clusters) + ' clusters')
        ax.view_init(30, 125)
        plt.show()

    return labels

def choose_random_cluster_samples(labels):
    # select 1 random sample from each cluster
    from collections import defaultdict
    # Initialize a dictionary to store the indices of each value
    indices = defaultdict(list)
    # Iterate over the elements and their indices in the list
    for i, value in enumerate(labels):
        # Append the index to the list of indices for the current value
        indices[value].append(i)
    # Iterate over the set of unique values in the list
    chosen_samples = []
    for value in set(labels):
        # Get a random index from the list of indices for the current value
        random_index = random.choice(indices[value])
        # print('Value:', value, 'Index:', random_index)
        chosen_samples.append(random_index)

    return chosen_samples

# # load and append all datasets into one list for ease of use
# regions = ["lagos_dataset",         # 0
#            "marmara_dataset",       # 1
#            "neworleans_dataset",    # 2
#            "venice_dataset",        # 3
#            "accra_dataset",         # 4
#            "durban_dataset"]        # 5
# datasets = []
# for i in range(len(regions)):
#     region = "datasets/" + regions[i] + ".pickle"
#     with open(region, 'rb') as data:
#         dataset = pickle.load(data)
#     datasets.append(dataset)
#
# feature_vectors_debris, feature_vectors_nondebris, feature_vectors_all, y_all = get_feature_vectors(datasets[5])
# pca_visualization(pca_components=3, feature_vectors_all=feature_vectors_all, y_all=y_all)
# cluster_all_features(feature_vectors_all, n_clusters=10, visualize=True)
