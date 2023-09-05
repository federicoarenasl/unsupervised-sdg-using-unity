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

PROG_PATH = 'progress'
GRAPH_PATH = 'graphs'
DIST_PATH = 'images'
FMAP_PATH = 'feature_maps'
TGT_PATH = 'images/target'
gen_PATH = '../data/datagen/scenes/unity/3d_scene_tgt/Builds/3dv1_Data/StreamingAssets/Snapshots'
GRAD_PATH = 'gradients'

# General helper functions
def import_npy(file_name):
    return np.load(file_name)

def import_json(file_name):
    with open(file_name, 'r') as f:
        config = json.load(f)
    
    return config

def import_csv(file_name):
    return pd.read_csv(file_name)

def import_png(file_name):
    img = cv2.imread(file_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def walk_npy_folder(directory):
    npys = []
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                npy = import_npy(os.path.join(directory, filename))
                npys.append(npy)
        
        return np.array(npys)
    
    else:
        return np.empty((1,1))

def walk_json_folder(directory):
    jsons = []
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                json_ = import_json(os.path.join(directory, filename))
                jsons.append(json_)
        
        return jsons
    
    else:
        return []

def walk_png_folder(directory, n_samples):
    pngs = []
    random_sample = random.randint(0, n_samples)
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                file_n = int(filename.split('.')[0])
                if file_n == random_sample: 
                    png = import_png(os.path.join(directory, filename))
                    pngs.append(png)
        
        return pngs
    
    else:
        return []

def read_yaml(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)

    return config

def read_hyperaparameters(experiment_n):
    hpams_path = os.path.join(ROOT_PATH, f'experiment{experiment_n}', 'hyperparameters', f'experiment{experiment_n}_hyperparameters.yaml')
    return read_yaml(hpams_path)

def pca_array(array, components=None):
    pca = PCA(n_components=components)
    return pca.fit_transform(array)

def jitter(values, std):
    return np.array(values) + np.random.normal(np.mean(values),std)

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range