from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import skimage.io
import skimage.transform
from skimage.io import imsave

import time

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

from visualization_utils import *

tsne = True
n_pca_dims = 100
n_tsne_iters = 10000
real_id = 100
sim_id = 101
info = {'real':real_id,'sim':sim_id} 

#get real data
real_image_file_root = '/Users/sarabeery/Documents/CameraTrapClass/Data/imerit_deer_ims/'
real_db_file = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/vizualize_activations/real_deer_visualization.json'
real_bbox_data = json.load(open(real_db_file,'r'))
real_cat_id_to_cat = {cat['id']:cat for cat in real_bbox_data['categories']}
real_ann_id_to_ann = {ann['id']:ann for ann in real_bbox_data['annotations']}

pickle_file = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/vizualize_activations/train_on_cct/real_deer.p'

real_data = pickle.load(open(pickle_file,'rb'))

print(real_data.keys())
real_activations = real_data['activations']
real_layer_names = real_data['layer_names']
real_labels = real_data['labels']
real_ann_ids = real_data['ids']
real_logits = real_data['logits']
real_paths = np.asarray([real_image_file_root+real_ann_id_to_ann[idx]['image_id']+'.jpg' for idx in real_ann_ids])
real_bboxes = np.asarray([real_ann_id_to_ann[idx]['bbox'] for idx in real_ann_ids])
print(real_data['logits'][0])
print(real_layer_names)

#get_sim_data
sim_image_file_root = '/Users/sarabeery/Documents/CameraTrapClass/Data/unity_sim_deer/'
sim_db_file = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/vizualize_activations/unity_deer_visualization.json'
sim_bbox_data = json.load(open(sim_db_file,'r'))
sim_cat_id_to_cat = {cat['id']:cat for cat in sim_bbox_data['categories']}
sim_ann_id_to_ann = {str(ann['id']):ann for ann in sim_bbox_data['annotations']}

pickle_file = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/vizualize_activations/train_on_cct/unity_deer.p'

sim_data = pickle.load(open(pickle_file,'rb'))

print(sim_data.keys())
sim_activations = sim_data['activations']
sim_layer_names = sim_data['layer_names']
sim_labels = sim_data['labels']
sim_ann_ids = sim_data['ids']
sim_logits = sim_data['logits']
sim_paths = np.asarray([sim_image_file_root+sim_ann_id_to_ann[idx]['image_id']+'.jpg' for idx in sim_ann_ids])
sim_bboxes = np.asarray([sim_ann_id_to_ann[idx]['bbox'] for idx in sim_ann_ids])
print(sim_data['logits'][0])
print(sim_layer_names)

layer_index = -1
#real
layer = real_activations[layer_index]
print(layer.shape)
image_idxs = range(len(real_labels))
real_X = []
for im_idx in image_idxs:
    vec = layer[im_idx]
    real_X.append(vec.flatten())
    
real_X = np.asarray(real_X)
real_y = np.asarray([100 for i in range(len(real_labels))])

print(real_X.shape, real_y.shape)

#sim
layer = sim_activations[layer_index]
print(layer.shape)
image_idxs = range(len(sim_labels))
sim_X = []
for im_idx in image_idxs:
    vec = layer[im_idx]
    sim_X.append(vec.flatten())
    
sim_X = np.asarray(sim_X)
sim_y = np.asarray([101 for i in range(len(sim_labels))])

print(sim_X.shape, sim_y.shape)

X = np.concatenate((real_X,sim_X))
y = np.concatenate((real_y,sim_y))
paths = np.concatenate((real_paths,sim_paths))
bboxes = np.concatenate((real_bboxes,sim_bboxes))

print(X.shape, y.shape, paths.shape, bboxes.shape)


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))
df['paths'] = paths
df['paths'] = df['paths'].apply(lambda i: str(i))

print('Size of the dataframe: {}'.format(df.shape))


X, y = None, None

plot_idxs = list(range(2000))

if not tsne:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    embedding = pca_result[plot_idxs,:2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
else:
    pca = PCA(n_components=n_pca_dims)
    pca_result = pca.fit_transform(df[feat_cols].values)
    print('Cumulative explained variation for 50 principal components:{}'.format(np.sum(pca.explained_variance_ratio_)))
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_tsne_iters, n_jobs=4)
    tsne_pca_results = tsne.fit_transform(pca_result)
    embedding = tsne_pca_results[plot_idxs,:2]

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

labels = df['label'][plot_idxs]
paths = df['paths'][plot_idxs]
plot_bboxes = bboxes[plot_idxs]

plot_embedding(embedding, labels, paths, plot_bboxes, info)

