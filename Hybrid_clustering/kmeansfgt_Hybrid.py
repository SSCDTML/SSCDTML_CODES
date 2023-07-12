from sklearn.cluster import KMeans
from skimage.color import rgb2gray
import numpy as np

def kmeansfgt(inMask, img, K, th):
    # Use Intensity-based clustering
    # Syntax:
    #     [outMask, imClusters, partialMasks] = kmeansfgt(inMask,I,K,th)
    # Inputs:
    #     inMask,         NxM Binary mask of density segmentation
    #     img,            NxM mammography image
    #     K,              Integer value for K clusters
    #     th,             Float value for Intensity Threshold
    # Outputs:
    #     outMask,        Binary mask of dense tissue segmentation
    #     imClusters,     For plotting the different clusters
    #     partialMasks,   For plotting the pixels within each cluster

    #img = rgb2gray(img) # convertir imagen RGB a escala de grises
    C = np.linspace(0, 1, K)
    kmeans = KMeans(n_clusters=K, init=C.reshape(-1, 1), n_init=1, max_iter=1000)
    kmeans.fit(img.reshape(-1, 1))
    outC = kmeans.labels_.reshape(img.shape)
    outMask = np.zeros_like(img)

    for i in range(K):

        cluster = outC == i
        partialMask = inMask & cluster
        measure = np.sum(partialMask.flatten()*1)/np.sum(cluster.flatten()*1)
       
        if measure >= th:
            
            outMask = outMask.astype(np.uint8)
            outMask = outMask | cluster # Acum clusters to final mask

    outMask = outMask & inMask
    imClusters = None  # since we don't use it
    partialMasks = None  # since we don't use it

    return outMask, imClusters, partialMasks