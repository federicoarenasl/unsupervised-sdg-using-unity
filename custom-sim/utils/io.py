"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import os
import json
from networkx.drawing.nx_pylab import draw_networkx
import yaml
import warnings
import shutil
import skimage.io as sio
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def generate_data(generator, out_dir, num_samples):
  """
  Generate data and save to an output directory

  generator: an object of the generator class
  out_dir: target to save data to
  num_samples: number of samples to generate
  """
  for i in tqdm(range(num_samples), desc='Generating data'):
    out_img = os.path.join(out_dir, f'{str(i).zfill(6)}.jpg')        
    out_lbl = os.path.join(out_dir, f'{str(i).zfill(6)}.json')
    # are you really making more than .zfill(6) can handle!
    g = generator.sample()
    r = generator.render(g)
    img, lbl = r[0]
    
    # write to disk
    write_img(img, out_img)
    write_json(lbl, out_lbl)

def makedirs(out_dir):
  if os.path.isdir(out_dir):
    #warnings.warn(f'Directory {out_dir} exists. Deleting!')
    shutil.rmtree(out_dir)

  os.makedirs(out_dir)

def makedirs_light(out_dir):
  if os.path.isdir(out_dir):
    warnings.warn(f'Directory {out_dir} exists.')
  else:
    os.makedirs(out_dir)

def create_dirs(opts):
  # Create image directories
  imagedir = os.path.join(opts['logdir'], opts['variant_name'], opts['imagedir'])
  tgt_imagedir = os.path.join(opts['logdir'], opts['variant_name'], opts['imagedir'], 'target')
  makedirs(imagedir)
  # Copy target data
  shutil.copytree(opts['task']['val_root'], tgt_imagedir)
  # Logdir for generated csv files
  csvdir = os.path.join(opts['logdir'], opts['variant_name'], opts['csvdir'])
  makedirs(csvdir)
  # Logdir for generated graph files
  graphdir = os.path.join(opts['logdir'], opts['variant_name'], opts['graphdir'])
  makedirs(graphdir)
  # Copy grammar file
  shutil.copyfile(opts['config'], os.path.join(graphdir, 'grammar.json'))

  # Logdir for saving weights
  wdir = os.path.join(opts['logdir'], opts['variant_name'], opts['weightsdir'])
  makedirs(wdir)
  # Logdir for saving gradients
  gradsdir = os.path.join(opts['logdir'], opts['variant_name'], opts['gradsdir'])
  makedirs(gradsdir)
  # Logdir for saving feature maps
  fmapsdir = os.path.join(opts['logdir'], opts['variant_name'], opts['fmapsdir'])
  makedirs(fmapsdir)
  # Logdir for saving hyperparameters
  hparamsdir = os.path.join(opts['logdir'], opts['variant_name'], opts['hparamsdir'])
  makedirs(hparamsdir)
  # Save hyperparameters and experiment description
  write_yaml(opts, os.path.join(hparamsdir,opts['variant_name']+"_hyperparameters.yaml"))
  write_txt(opts['description'], os.path.join(hparamsdir,opts['variant_name']+"_description.txt"))

  return imagedir, csvdir, graphdir, wdir, gradsdir, fmapsdir, hparamsdir

def read_json(fname):
  with open(fname, 'r') as f:
    config = json.load(f)
    
  return config

def write_json(config, fname):
  with open(fname, 'w') as f:
    json.dump(config, f)

def read_yaml(fname):
  with open(fname, 'r') as f:
    config = yaml.safe_load(f)

  return config

def write_yaml(opts, fname):
  with open(fname, 'w') as f:
    yaml.dump(opts, f)

def write_txt(description, fname):
  with open(fname, 'w') as f:
    f.write(description)

def inspect_img(img):
  plt.axis('off')
  plt.grid(b=None)
  plt.title('')
  plt.legend('',frameon=False)
  plt.imshow(img.permute(1, 2, 0))
  plt.show()

def write_img(img, fname):
  sio.imsave(fname, img)

def write_csv(data_dict, dir_name, fname):
  df = pd.DataFrame.from_dict(data_dict)
  out_name = os.path.join(dir_name, fname)
  df.to_csv(out_name, index=None)

def plot_progress(data_dict, dir_name, fname):
  out_name = os.path.join(dir_name, fname)
  df = pd.DataFrame.from_dict(data_dict)
  sns.set_style("whitegrid")
  fig, ax = plt.subplots(figsize=(12,6))
  plt.plot(df['rec_epoch'], df['class_loss'], label='class loss')
  plt.plot(df['rec_epoch'], df['cont_loss'], label='cont loss')
  plt.plot(df['rec_epoch'], df['loss'], label='(total) loss')
  plt.legend(fontsize=12)
  plt.title("GCN train loss", fontweight='bold', fontsize=14)
  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("Losses", fontsize=12)
  plt.savefig(out_name, bbox_inches='tight')

def plot_mmd(data_dict, dir_name, fname, epoch):
  out_name = os.path.join(dir_name, fname)
  df = pd.DataFrame.from_dict(data_dict)
  sns.set_style("whitegrid")
  fig, ax = plt.subplots(figsize=(12,6))
  plt.plot(df['epoch'], df['mmd_loss'], label='MMD loss')
  plt.legend(fontsize=12)
  plt.title(f"MMD loss at epoch {epoch}", fontweight='bold', fontsize=14)
  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("Loss", fontsize=12)
  plt.savefig(out_name, bbox_inches='tight')

def plot_task_loss(data_dict, dir_name, fname, epoch):
  out_name = os.path.join(dir_name, fname)
  df = pd.DataFrame.from_dict(data_dict)
  sns.set_style("whitegrid")
  fig, ax = plt.subplots(figsize=(12,6))
  plt.plot(df['epoch'], df['task_loss'], label='Task loss')
  plt.legend(fontsize=12)
  plt.title(f"Task loss at epoch {epoch}", fontweight='bold', fontsize=14)
  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("Loss", fontsize=12)
  plt.savefig(out_name, bbox_inches='tight')

def plot_acc(data_dict, dir_name, fname, epoch):
  out_name = os.path.join(dir_name, fname)
  df = pd.DataFrame.from_dict(data_dict)
  sns.set_style("whitegrid")
  fig, ax = plt.subplots(figsize=(12,6))
  plt.plot(df['epoch'], df['task_acc'], label='Accuracy')
  plt.legend(fontsize=12)
  plt.title(f"Accuracy at epoch {epoch}", fontweight='bold', fontsize=14)
  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("Acc", fontsize=12)
  plt.savefig(out_name, bbox_inches='tight')

def movingaverage(interval, window_size):
  window = np.ones(int(window_size))/float(window_size)
  return np.convolve(interval, window, 'same')

def plot_graph(G, out_graph):
  plt.figure(figsize=(10,12))
  pos = nx.shell_layout(G)
  node_labels = prepare_graph_labels(G)
  nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 8000, label=node_labels)
  nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)
  nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True, arrowsize=300, connectionstyle='arc3,rad=0.5')
  plt.savefig(out_graph, bbox_inches='tight')

def write_graph_json(G, out_graph):
  dictio = nx.readwrite.json_graph.node_link_data(G)

  for i, node in enumerate(dictio['nodes']):
    for attr in node['attr'].keys():
      if type(node['attr'][attr]) == np.ndarray:
        list_array = node['attr'][attr].tolist()
        dictio['nodes'][i]['attr'][attr] = list_array

  with open(out_graph, 'w') as outfile1:
    outfile1.write(json.dumps(dictio))

def prepare_graph_labels(G):
  labels = {}
  mutable_attrs = G._node[0]['attr']['mutable']
  for i in G:
    class_label = str(i)+":"+str(G._node[i]['cls'])
    if not G._node[i]['attr']['immutable']:
      attr_label = ''
      for mutable_attr in mutable_attrs:
        attr_label = attr_label+'\n'+mutable_attr+':'+str(round(G._node[i]['attr'][mutable_attr],2))
      class_label = class_label +'\nattrs:'+'['+attr_label+']'
    labels[i] = class_label
  
  return labels

def write_npy(array, fname):
  with open(fname, 'wb') as f:
    np.save(f, array)

def output_img_lbl(img, lbl, out_dir, i, format):
  out_img = os.path.join(out_dir, f'{str(i).zfill(6)}.{format}')        
  out_lbl = os.path.join(out_dir, f'{str(i).zfill(6)}.json')
  write_img(img, out_img)
  write_json(lbl, out_lbl)

def output_graph_feats(g, out_dir, i):
  out_graph = os.path.join(out_dir, f'{str(i).zfill(6)}.npy')  
  with open(out_graph, 'wb') as f:
    np.save(f, g)

def output_recent_data(im_dir, graph_dir, img, lbl, f, i, format):
  output_img_lbl(img, lbl, im_dir, i, format)
  output_graph_feats(f, graph_dir, i)

def output_epoch_data(imagedir, graphdir, e, img, lbl, f, i, format):
  epoch_im_dir = os.path.join(imagedir, f'epoch_{e}')
  epoch_graph_dir = os.path.join(graphdir, f'epoch_{e}')
  makedirs_light(epoch_im_dir)
  makedirs_light(epoch_graph_dir)
  output_img_lbl(img, lbl, epoch_im_dir, i, format)
  output_graph_feats(f, epoch_graph_dir, i)

def save_gradients(epoch_gradients, parameters, idx):
  epoch_gradients['layer'] = []
  epoch_gradients[f"batch_{idx}"] = []
  for name, param in parameters:
    epoch_gradients['layer'].append(name)
    epoch_gradients[f"batch_{idx}"].append(param.grad.norm().cpu().item())
  
  return epoch_gradients

def save_epoch_weights(e, wdir, model, optimizer, mmd, optimizer_mmd=None):
  last_weights = os.path.join(wdir, f"weights_last.pt")
  optim_mmd_state_dict = optimizer_mmd.state_dict() if optimizer_mmd else None
  torch.save({
      'epoch': e,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'mmd_model_state_dict': mmd.model.state_dict(),
      'mmd_optimizer_state_dict': optim_mmd_state_dict,
      }, last_weights)


def export_gradients(e, opts, egrads, gradsdir, egrads_inception, inception_gradsdir):
  # Output epoch's gradients every 5 epochs
  if e%5==0:
    if opts['optim_mmd']['use_mmd_loss']:
      write_csv(egrads_inception, inception_gradsdir, f'grads_e_{e}.csv')
    else:
      write_csv(egrads, gradsdir, f'grads_e_{e}.csv')
  
def save_mmd_progress(e, mmd, mmd_train_prog, egen_time, etime, csvdir):
  # Append MMD train progress
  mmd_train_prog['epoch'].append(e)
  mmd_train_prog['mmd_loss'].append(mmd.item())
  mmd_train_prog['dataset_gen_time'].append(egen_time)
  mmd_train_prog['epoch_runtime'].append(etime)
  # Output train logs
  write_csv(mmd_train_prog, csvdir, 'mmd.csv')
  plot_mmd(mmd_train_prog, csvdir, 'mmd.png', e)

  return mmd_train_prog

def save_mmd_progress_mnist(e, mmd, mmd_train_prog, task_loss, loss, acc, egen_time, etime, csvdir):
  # Append MMD train progress
  mmd_train_prog['epoch'].append(e)
  mmd_train_prog['mmd_loss'].append(mmd.item())
  mmd_train_prog['dataset_gen_time'].append(egen_time)
  mmd_train_prog['epoch_runtime'].append(etime)
  mmd_train_prog['task_loss'].append(task_loss)
  mmd_train_prog['loss'].append(loss)
  mmd_train_prog['task_acc'].append(acc)

  # Output train logs
  write_csv(mmd_train_prog, csvdir, 'mmd_mnist.csv')
  plot_mmd(mmd_train_prog, csvdir, 'mmd.png', e)
  plot_task_loss(mmd_train_prog, csvdir, 'task_loss.png', e)
  plot_acc(mmd_train_prog, csvdir, 'accuracy.png', e)

  return mmd_train_prog
  

def export_graph(G, graph_dir,  i, gt=True):
  # Loop through every graph in the batch
  for k in range(len(G)):
    # If generating ground truth graphs
    if gt:
      gout_graph = os.path.join(graph_dir, f'gtgraph_{str(i).zfill(6)}.jpg')      
      gout_graph_json = os.path.join(graph_dir, f'gtgraph_{str(i).zfill(6)}.json')      
      i+=1      
    # If generating graphs generated by GCN
    else:
      gout_graph = os.path.join(graph_dir, f'ggraph_{str(i).zfill(6)}.jpg')      
      gout_graph_json = os.path.join(graph_dir, f'ggraph_{str(i).zfill(6)}.json')
      i+=1 
    # Save graph as plot and as json, comment plot_graph for mass data generation and increased speed
    #io.plot_graph(G[k], gout_graph)
    write_graph_json(G[k], gout_graph_json)
  
  return i


def export_feature_maps(fmaps_dir, tgt_fmaps, gen_fmaps, idx, e):
  for i in range(len(tgt_fmaps)):
    tgt_fmap = tgt_fmaps[i].cpu().detach().numpy()
    gen_fmap = gen_fmaps[i].cpu().detach().numpy()
    dim = tgt_fmap.shape[-1]
    outputdir = os.path.join(fmaps_dir, f"fmaps_{dim}") 
    makedirs_light(outputdir)
    write_npy(tgt_fmap, os.path.join(outputdir, f'fmap{i}_tgt_batch_{idx}_dim_{dim}.npy'))
    write_npy(gen_fmap, os.path.join(outputdir, f'fmap{i}_gen_batch_{idx}_dim_{dim}.npy'))

