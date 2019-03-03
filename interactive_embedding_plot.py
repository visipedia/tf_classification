from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import skimage.io
import skimage.transform
from skimage.io import imsave

sys.path.append('/Users/sarabeery/Documents/CameraTrapClass/code/CameraTraps/database_tools/')
sys.path.append('/Users/sarabeery/Documents/CameraTrapClass/code/CameraTraps/classification_eval/')


from data_loader_coco_format import COCODataLoader
from results_loader import ResultsLoader
from evaluate_cropped_box_classification import *
from visualization_utils import *

import time

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA


def get_X(result_list,layer_idx,):
    X = np.array([])
    for result in result_list:
        new_X = result.get_flattened_layer_activations(layer_idx)
        X = np.concatenate((X,new_X)) if X.size > 0 else new_X


    return X

def get_y(result_list,label_addition_list):
    y = np.array([])
    for idx, result in enumerate(result_list):
        label_addition = label_addition_list[idx]
        #print(set([i for i in result.labels]))
        new_y = np.asarray([i+label_addition for i in result.labels])
        y = np.concatenate((y,new_y)) if y.size > 0 else new_y

    return y



def main():

    load_from_embedding_file = False
    tsne = True
    n_pca_dims = 100
    n_tsne_iters = 15000
    real_id = 51
    sim_id = 50
    layer_index = -1
    file_root = '/Users/sarabeery/Documents/CameraTrapClass/'
    vis_folder = file_root+'sim_classification/vizualize_activations/'
    cct_db_file = file_root+'/Fixing_CCT_Anns/Corrected_versions/CombinedBBoxAndECCV18.json'
    category_file = file_root+'sim_classification/eccv_categories.json'
    experiment = 'unity_300K_w_deer/'

    embedding_save_file = vis_folder+experiment+'tsne_embedding.p'

    plot_save_name = vis_folder+experiment+'embedding_plot.jpg'

    if not load_from_embedding_file:

        #get real data

        real_image_file_root = file_root+'Data/imerit_deer_ims/'
        real_db_file = cct_db_file#vis_folder+'real_deer_visualization.json'
        real_results_file = vis_folder+experiment+'real_deer.p'

        real_data = load_database(real_db_file,real_image_file_root,
            alternate_category_file=category_file)
        real_results = load_results(real_results_file, 
            image_file_root=real_image_file_root, database=real_data)

        print('loaded real data')

        #get_sim_data
        sim_image_file_root = file_root+'Data/unity_sim_deer/'
        sim_db_file = vis_folder+'/unity_deer_visualization.json'
        sim_results_file = vis_folder+experiment+'unity_deer.p'

        sim_data = load_database(sim_db_file,sim_image_file_root,
            alternate_category_file=category_file)
        sim_results = load_results(sim_results_file, 
            image_file_root=sim_image_file_root, database=sim_data)
        print(set(sim_results.labels))

        print('loaded sim data')
        #get_cis_val_data

        cis_image_file_root = file_root+'Data/cis_val/'
        cis_db_file = vis_folder+'cis_val_visualization.json'
        cis_results_file = cct_db_file#vis_folder+experiment+'cis_val.p'

        cis_data = load_database(cis_db_file,cis_image_file_root,
            alternate_category_file=category_file)
        cis_results = load_results(cis_results_file, 
            image_file_root=cis_image_file_root, database=cis_data)

        print('loaded cis data')

        trans_image_file_root = file_root+'Data/trans_val/'
        trans_db_file = cct_db_file#vis_folder+'trans_val_visualization.json'
        trans_results_file = vis_folder+experiment+'trans_val.p'

        trans_data = load_database(trans_db_file,trans_image_file_root,
            alternate_category_file=category_file)
        trans_results = load_results(trans_results_file, 
            image_file_root=trans_image_file_root, database=cis_data)

        print('loaded trans data')

        #make info dict
        info = {cat['name']:cat['id'] for cat in real_data.categories}
        for cat in info:
            info['trans_'+cat] = real_id + info[cat]
        info['sim_deer'] = sim_id+list(set(sim_results.labels))[0]
        #info['real_trans_deer'] = real_id+list(set(real_results.labels))[0]

        result_list = [real_results,sim_results,cis_results,trans_results]
        label_addition_list = [real_id,sim_id,0,real_id]

        #real
        X = get_X(result_list,layer_index)
        y = get_y(result_list,label_addition_list)

        paths = np.concatenate((real_results.paths,
            sim_results.paths,cis_results.paths))
        bboxes = np.concatenate((real_results.bboxes,
            sim_results.bboxes,cis_results.bboxes))
        correct = np.concatenate((real_results.correct,
            sim_results.correct,cis_results.correct))

        print(X.shape, y.shape, paths.shape, bboxes.shape)


        feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

        df = pd.DataFrame(X,columns=feat_cols)
        df['label'] = y
        df['label'] = df['label'].apply(lambda i: str(i))
        df['paths'] = paths
        df['paths'] = df['paths'].apply(lambda i: str(i))

        print('Size of the dataframe: {}'.format(df.shape))

        plot_idxs = list(range(len(y)))
        print(len(y))

        X, y = None, None



        if not tsne:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df[feat_cols].values)

            embedding = pca_result[plot_idxs,:2]

            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
        else:
            pca = PCA(n_components=n_pca_dims)
            pca_result = pca.fit_transform(df[feat_cols].values)
            print('Cumulative explained variation for {} principal components:{}'.format(n_pca_dims,np.sum(pca.explained_variance_ratio_)))
    
            time_start = time.time()
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_tsne_iters, n_jobs=4)
            tsne_pca_results = tsne.fit_transform(pca_result)
            embedding = tsne_pca_results[plot_idxs,:2]

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        labels = df['label'][plot_idxs]
        paths = df['paths'][plot_idxs]
        bboxes = bboxes[plot_idxs]
        correct = correct[plot_idxs]

        pickle.dump({'labels':labels,'paths':paths,'bboxes':bboxes,
            'embedding':embedding, 'info':info, 'correct': correct},
            open(embedding_save_file,'w'))

        #np.savez(embedding_save_file,labels=labels,paths=paths,
        #    bboxes=bboxes,embedding=embedding,info=info)
    else:
        with open(embedding_save_file,'rb') as df:
            f = pickle.load(df)
            labels = f['labels']
            paths = f['paths']
            bboxes = f['bboxes']
            embedding = f['embedding']
            info=f['info']
            correct = f['correct']

    print(info['deer'])
    plot_embedding(embedding, labels, paths, bboxes, info, correct=correct)

    save_embedding_plot(plot_save_name, embedding, labels, info)

if __name__ == '__main__':
    main()

