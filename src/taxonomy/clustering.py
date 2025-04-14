from scipy.cluster.vq import kmeans as kmeans
from sklearn.cluster import KMeans

def kmeans_clustering(observations, centroids=10):
    if centroids > len(observations):
        print('WARNING: you have less observations than centroids')
        centroids = len(observations)
    kmeans = KMeans(n_clusters=centroids)
    centers = kmeans.fit(observations).fit_predict(observations) #kmeans(observations, centroids)
    centers = [int(center) for center in centers] # making this list json serialize
    return centers

# 
def binary_encoding(observations):
    pass 
    #return binary_mapping
