import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import os

if 'DISPLAY' not in os.environ.keys():
    plt.switch_backend('Agg')


def distn(p1, p2, degree):
    return euclidean_distances(np.diff(p1, n=degree, axis=1), np.diff(p2, n=degree, axis=1))

def customargmaxmin(vector, size, max):
    if (max == 1):
        if ( size[vector[0]] > size[vector[1]]):
            return vector[0]
        else:
            return vector[1]
    else:
        if ( size[vector[0]] > size[vector[1]]):
            return vector[1]
        else:
            return vector[0]

def recurssiveclustering(centroids, imageLabels, cluster_size, labels, n_clusters, model = None, posinfo = None, data = None):
    centroidsDistances = distn(centroids, centroids, 0) + 10000*np.eye(centroids.shape[0])
    updatedLabels = imageLabels
    labelslist = labels
    updatedCentroids = centroids
    numberOfClusters = centroids.shape[0]
    while(numberOfClusters > n_clusters):
        mixLabels = np.argwhere(centroidsDistances == centroidsDistances.min())
        mixLabels = mixLabels[0, :]     # Matriu Simetrica
        majorClusterIndex = customargmaxmin(mixLabels, cluster_size, 1)
        minorClusterIndex = customargmaxmin(mixLabels, cluster_size, 0)
        if (((cluster_size[minorClusterIndex] > 0) and (cluster_size[majorClusterIndex] > 0)) or (minorClusterIndex != majorClusterIndex)):
            newClusterSize = cluster_size[majorClusterIndex] + cluster_size[minorClusterIndex]
            updatedCentroids[majorClusterIndex] = (updatedCentroids[majorClusterIndex] * (
                        cluster_size[majorClusterIndex] / newClusterSize)) + (updatedCentroids[minorClusterIndex] * (
                                                         cluster_size[minorClusterIndex] / newClusterSize))
            updatedCentroids = np.delete(updatedCentroids, minorClusterIndex, axis=0)
            cluster_size[majorClusterIndex] = newClusterSize
            cluster_size = np.delete(cluster_size, minorClusterIndex)
            for j in range(updatedLabels.shape[0]):
                if (updatedLabels[j] == labelslist[minorClusterIndex]):
                    updatedLabels[j] = labelslist[majorClusterIndex]
            labelslist = np.delete(labelslist, minorClusterIndex)
            numberOfClusters -= 1
            centroidsDistances = distn(updatedCentroids, updatedCentroids, 0) \
                                 + 10000 * np.eye(updatedCentroids.shape[0])
    return updatedCentroids, updatedLabels, cluster_size

def generatecurvespng(representatives, dataSet, labels, imagelabels, x_axis=None, x_axis_name=None):

    rows, cols = int(labels.shape[0]/2), 2
    if x_axis is None:
        x_axis = np.arange(0,representatives.shape[1])

    y_lim_min = np.min(dataSet)
    y_lim_max = np.max(dataSet)
    varianceplot = np.zeros_like(representatives)
    f, axarr = plt.subplots(rows,cols, sharex=True)
    for i in range(labels.shape[0]):
        auxlist = np.where(imagelabels == labels[i])[0]
        # varianceplot[i] = np.mean(np.power(dataSet[auxlist]-np.tile(representatives[i], (dataSet[auxlist].shape[0], 1)), 2))
        curves = dataSet[auxlist]

        if rows == 1:
            it_col = np.remainder(i, 2)
            for k in range(curves.shape[0]):
                axarr[it_col].plot(x_axis, curves[k, :], 'r')
            axarr[it_col].plot(x_axis, [0] * x_axis.shape[0], 'b')
            axarr[it_col].plot(x_axis, representatives[i], 'k')
            axarr[it_col].set_ylim(y_lim_min, y_lim_max)
            if x_axis_name is not None:
                axarr[it_col].set_xlabel(x_axis_name)
        else:
            it_row, it_col = int(i/2), np.remainder(i,2)
            for k in range(curves.shape[0]):
                axarr[it_row, it_col].plot(x_axis, curves[k, :], 'r')
            axarr[it_row, it_col].plot(x_axis, [0]*x_axis.shape[0], 'b')
            axarr[it_row, it_col].plot(x_axis, representatives[i], 'k')
            axarr[it_row, it_col].set_ylim(y_lim_min, y_lim_max)
            axarr[it_row, it_col].set_title(labels[i]+1)
            if x_axis_name is not None and it_row == rows - 1:
                axarr[it_row, it_col].set_xlabel(x_axis_name)
    return f

def generate_brain_labels(labels, posinfo, template_array):
    image_labels = np.zeros_like(template_array)
    for i in range(posinfo.shape[0]):
        image_labels[posinfo[i, 0].astype(int), posinfo[i, 1].astype(int), posinfo[i, 2].astype(int)] = labels[i]+1
    return image_labels


class HierarchicalClustering(object):

    def __init__(self, n_clusters, template):
        self.n_clusters = n_clusters
        self.template = template
        self._name = 'hierarchical'

    def clusterize(self, curves, index_matrix, x_axis=None, x_axis_name=None):

        n_curves = curves.shape[1]
        precomputedDistances = 0.2 * distn(curves, curves, 0) + 0.8 * distn(curves, curves, 1) + \
                               0.4 * distn(curves, curves, 2)

        agglomerative = AgglomerativeClustering(affinity='precomputed',
                                                linkage='complete',
                                                n_clusters=self.n_clusters).fit(precomputedDistances)

        representatives = np.zeros((self.n_clusters, n_curves))
        cluster_size = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            auxlist = np.where(agglomerative.labels_ == i)[0]
            representatives[i] = np.mean(curves[auxlist], axis=0)
            cluster_size[i] = auxlist.shape[0]

        image_labels = generate_brain_labels(agglomerative.labels_, index_matrix, self.template)
        clustering_name = 'clusteringlabels_' + str(self.n_clusters) + '_' + self._name
        results = [(clustering_name, image_labels)]

        png_figure = generatecurvespng(representatives=representatives, dataSet=curves,
                                       labels=np.unique(agglomerative.labels_), imagelabels=agglomerative.labels_,
                                       x_axis=x_axis, x_axis_name=x_axis_name)
        png_name = 'clusteringcurves_' + str(self.n_clusters) + '_' + self._name
        png_tuple = [(png_name, png_figure)]

        return results, png_tuple


class RecursiveClustering(object):
    def __init__(self, n_clusters, template, n_clusters_hierarchical = 20):
        self.n_clusters = n_clusters
        self.template = template
        self._name = 'recursive'
        self.n_clusters_hierarchical = n_clusters_hierarchical


    def clusterize(self, curves, index_matrix, x_axis=None, x_axis_name=None):


        n_curves = curves.shape[1]
        precomputedDistances = 0.2 * distn(curves, curves, 0) + 0.8 * distn(curves, curves, 1) + \
                               0.2 * distn(curves, curves, 2)

        agglomerative = AgglomerativeClustering(affinity='precomputed',
                                                linkage='complete',
                                                n_clusters=self.n_clusters_hierarchical).fit(precomputedDistances)

        representatives = np.zeros([self.n_clusters_hierarchical, n_curves])
        cluster_size = np.zeros(self.n_clusters_hierarchical)

        for i in range(self.n_clusters_hierarchical):
            auxlist = np.where(agglomerative.labels_ == i)[0]
            representatives[i] = np.mean(curves[auxlist], axis=0)
            cluster_size[i] = auxlist.shape[0]

        representatives, definitiveLabels, cluster_size = recurssiveclustering(centroids=representatives,
                                                                               imageLabels=agglomerative.labels_,
                                                                               cluster_size=cluster_size,
                                                                               labels=np.unique(agglomerative.labels_),
                                                                               n_clusters=self.n_clusters)

        image_labels = generate_brain_labels(definitiveLabels, index_matrix, self.template)
        clustering_name = 'clusteringlabels_' + str(self.n_clusters) + '_' + self._name  + '_' + str(self.n_clusters_hierarchical)
        results = [(clustering_name, image_labels)]


        print(np.unique(definitiveLabels))
        png_figure = generatecurvespng(representatives=representatives, dataSet=curves,
                                       labels=np.unique(definitiveLabels), imagelabels=definitiveLabels,
                                       x_axis=x_axis, x_axis_name=x_axis_name)
        png_name = 'clusteringcurves_' + str(self.n_clusters) + '_' + self._name + '_' + str(self.n_clusters_hierarchical)
        png_tuple = [(png_name, png_figure)]

        return results, png_tuple