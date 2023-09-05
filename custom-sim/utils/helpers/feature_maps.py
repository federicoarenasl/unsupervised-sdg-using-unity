import pandas as pd
import cv2
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from tqdm import tqdm_notebook as tqdm

# Feature maps helper functions
def TSNE_fmaps(n_components, concat_array):
    components=n_components
    embedded = TSNE(n_components=components).fit_transform(concat_array).T

    return embedded

def get_e_fmaps_inception(fmap_folder, epoch_n, n_components):
    fmap_epoch_folder = os.path.join(fmap_folder, f'epoch_{epoch_n}', 'fmaps_2048')
    gen_tgt_dict = {'gen':[], 'tgt':[]}
    if os.path.isdir(fmap_epoch_folder):
        for filename in os.listdir(fmap_epoch_folder):
            if filename.endswith('.npy'):
                splitfname = filename.split('_')
                tgt_gen = splitfname[1]
                fmap_n = splitfname[0][-1]
                if tgt_gen == 'gen':
                    npy = import_npy(os.path.join(fmap_epoch_folder, filename))
                    gen_tgt_dict['gen'].append(npy)
                else:
                    npy = import_npy(os.path.join(fmap_epoch_folder, filename))
                    gen_tgt_dict['tgt'].append(npy)
        
        # Concatenate collected fmaps
        gen_tgt_dict['gen'] = TSNE_fmaps(n_components, np.concatenate(gen_tgt_dict['gen']))
        gen_tgt_dict['tgt'] = TSNE_fmaps(n_components, np.concatenate(gen_tgt_dict['tgt']))
    
    else:
        pass
    
    return gen_tgt_dict

def get_epoch_fmaps(experiment_n, interv, n_components):
    fmap_folder = os.path.join(ROOT_PATH, f'experiment{experiment_n}', FMAP_PATH)
    hpams = read_hyperaparameters(experiment_n)
    total_epochs = hpams['max_epochs']
    freq = hpams['distprint_freq']
    interval = interv

    fmaps_dict = {}
    for e in range(0, total_epochs+1, freq*interval):
        fmaps = get_e_fmaps_inception(fmap_folder, e, n_components)
        if type(fmaps['gen']) != list:
            fmaps_dict[f'epoch_{e}'] = fmaps
            
        else:
            print("Full results not ready yet")
            break

    return fmaps_dict

def plot_feature_representation(experiment, components, interval, filename, title):
    # Get all graph features in 3D representation
    fmap_features = get_epoch_fmaps(experiment, interval, components)
    # Creating figure
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 2, "ytick.major.size": 2})

    dim = len(fmap_features)
    fig = plt.figure(figsize = (dim*4, (dim*4)/dim))
    plt.suptitle(title, fontweight="bold", y=1.03, fontsize=20)
    for i, epoch in enumerate(fmap_features.keys()):
        # Create figure
        if components == 3:
            ax = fig.add_subplot(1, dim, i+1, projection='3d')
            # Get components
            x_tgt, y_tgt, z_tgt = fmap_features[epoch]['tgt'][0],fmap_features[epoch]['tgt'][1],fmap_features[epoch]['tgt'][2]
            x_gen, y_gen, z_gen = fmap_features[epoch]['gen'][0],fmap_features[epoch]['gen'][1],fmap_features[epoch]['gen'][2]

            std_jitter = 0.001
            # Creating plot
            ax.scatter3D(jitter(x_tgt, std_jitter), jitter(y_tgt, std_jitter), jitter(z_tgt,std_jitter) , color = "green", label='Target', alpha=0.8, s=25, linewidths=0.4, edgecolor='white')
            ax.scatter3D(jitter(x_gen,std_jitter), jitter(y_gen,std_jitter), jitter(z_gen,std_jitter) , color = "blue", label='Generated', alpha=0.8, s=25, linewidths=0.4, edgecolor='white')
            plt.title(f"Feature maps\nfrom {' '.join(epoch.split('_'))}", fontsize=16)
            if  i ==1:
                plt.legend(fontsize=16,loc="lower right")

        else:
            # Get components
            x_tgt, y_tgt= fmap_features[epoch]['tgt'][0],fmap_features[epoch]['tgt'][1]
            x_gen, y_gen= fmap_features[epoch]['gen'][0],fmap_features[epoch]['gen'][1]
            # Normalize components
            x_tgt, y_tgt = scale_to_01_range(x_tgt), scale_to_01_range(y_tgt)
            x_gen, y_gen = scale_to_01_range(x_gen), scale_to_01_range(y_gen)
            std_jitter = 0.0001
            # Creating plot
            ax = fig.add_subplot(1, dim, i+1)
            ax.scatter(jitter(x_tgt, std_jitter), jitter(y_tgt, std_jitter), color = "green", label='Target', alpha=0.25)
            ax.scatter(jitter(x_gen,std_jitter), jitter(y_gen,std_jitter), color = "blue", label='Generated', alpha=0.25)
            plt.title("Feature maps from "+' '.join(epoch.split('_')), fontsize=17)
            plt.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, f'{filename}.pdf'), bbox_inches='tight', pad_inches=0.2)