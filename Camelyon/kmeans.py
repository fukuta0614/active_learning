from sklearn.cluster import KMeans
import numpy as np
import pickle
feat = np.load('train_cbp512_feat.npy')
kmeans = KMeans(n_clusters=1000, random_state=0, n_init=5).fit(feat)

with open('kmeans_cache.pkl', 'wb') as f:
    pickle.dump(kmeans, f)