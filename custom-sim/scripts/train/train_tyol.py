"""
Adapted from:
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import argparse
from tqdm import tqdm

import utils
import utils.io as io
from data.loaders import get_loader, get_scene_graph_loader
from models.tasknet import get_tasknet
from models.metasim import MetaSim
from models.layers.render import RenderLayer
from models.layers.mmd import MMDInception
import time
import shutil

class Trainer(object):
  def __init__(self, opts):
    '''
    opts: parsed arguments from YAML experiment configurations

    This calss takes opts as input and fully trains the Meta-Sim architecture
    '''
    print("Initializing network...")
    # Import configurations from YAML file
    self.opts = opts
    self.device = opts['device']

    # Create all relevant directories to log training info
    self.imagedir, self.csvdir, self.graphdir, \
      self.wdir, self.gradsdir, self.fmapsdir, self.hparamsdir = io.create_dirs(self.opts)

    # Set seeds
    rn = utils.set_seeds(self.opts['seed'])
    # Set last epoch to 0
    self.last_epoch = 0

    # Initalize MetaSim model, which mainly contains the GCN
    dropout = None if self.opts['dropout'] == 'None' else self.opts['dropout']
    self.model = MetaSim(self.opts, dropout).to(self.device)
    self.generator = self.model.generator

    # Initialize data loaders
    self.init_loaders(self.opts)

    # Tasknet definition
    tasknet_class = get_tasknet(self.opts['dataset']) # We'll have to define a tasknet that accepts the TYOL dataset in due time
    self.tasknet = tasknet_class(self.opts['task']).to(self.opts['task']['device'])

    # Initialize rendering layer, this renderer includes updating the graph structure as we render
    self.renderer = RenderLayer(self.generator, self.device)

    # Initialize Maximum Mean Discrepancy metric, including the InceptionV3 network
    self.mmd = MMDInception(device=self.device, resize_input=self.opts['mmd_resize_input'], 
                              include_image=False, dims=self.opts['mmd_dims'])

    # Initialize Adam optimizer for GCN
    self.optimizer = torch.optim.Adam(
      self.model.parameters(),
      lr = opts['optim']['lr'],
      weight_decay = opts['optim']['weight_decay']
    )
    # Initialize Learning Rate scheduler for GCN
    self.lr_sched = torch.optim.lr_scheduler.StepLR(
      self.optimizer,
      step_size = opts['optim']['lr_decay'],
      gamma = opts['optim']['lr_decay_gamma']
    )

    # Optimize Inception network
    if opts['optim_mmd']['use_mmd_loss']:
      # Initialize Adam optimizer for MMD
      self.optimizer_mmd = torch.optim.Adam(
        self.mmd.model.parameters(),
        lr = opts['optim_mmd']['lr'],
        weight_decay = opts['optim_mmd']['weight_decay']
      )
      # Initialize Learning Rate scheduler for GCN
      self.lr_sched_mmd = torch.optim.lr_scheduler.StepLR(
        self.optimizer,
        step_size = opts['optim_mmd']['lr_decay'],
        gamma = opts['optim_mmd']['lr_decay_gamma']
      )

    # Load pre-trained weights if necessary
    if opts['load_weights']:
      checkpoint = torch.load(os.path.join(self.wdir, f"weights_last.pt"))
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.mmd.model.load_state_dict(checkpoint['mmd_model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.optimizer_mmd.load_state_dict(checkpoint['optimizer_mmd_state_dict'])
      self.last_epoch = checkpoint['epoch']
  
  def init_loaders(self, opts):
    # Initialize graph generator as a dataset sampled from probabilistic grammar
    sgl = get_scene_graph_loader(opts['dataset']) 
    self.scene_graph_dataset = sgl(self.generator, self.opts['epoch_length'])
    # Initialize target dataset for tasknet evaluation
    dl = get_loader(opts['dataset'])
    self.target_dataset = dl(self.opts['task']['val_root'])
    # Initialize target dataset for MMD calculation
    self.mmd_target_dataset = dl(self.opts['task']['val_root'])

  def train_reconstruction(self):
    '''
    This method pre-trains the Graph Convolutional Network for graph reconstruction
    '''
    # Initialize a loader for the batches of graphs sampled from the probabilistic grammar
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)
    # Initialize dictionary where we store training logs
    gcn_train_prog = {'rec_epoch':[], 'batch':[], 'class_loss':[], 'cont_loss':[], 'loss':[]}

    # Start training for reconstruction_epochs epochs
    for e in range(self.opts['reconstruction_epochs']):
      # For every batch of graphs in the loader
      for idx, (g, x, m, adj) in enumerate(loader):
        # g: scene graph, x: encoded features, m : mutability mask, adj: adjacency matrix
        x, m, adj = x.float().to(self.device), m.float().to(self.device),\
          adj.float().to(self.device)

        # Get generated/decoded graphs from GCN
        dec, dec_act = self.model(x, adj) 

        # Get classification log probabilities
        cls_log_prob = F.log_softmax(dec[..., :self.model.num_classes], dim=-1)

        # Get Negative Log Likelihood for the class loss
        cls_loss = -torch.mean(torch.sum(cls_log_prob * x[..., :self.model.num_classes], dim=-1))
        cls_loss *= self.opts['weight']['class']

        # Get Reconstruction loss (Mean Squared Error) distance between both reconstructions
        cont_loss = F.mse_loss(dec_act[..., self.model.num_classes:], 
          x[..., self.model.num_classes:])

        # Sum up both losses
        loss = cls_loss + cont_loss
        
        # Update GCN parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Print training progress
        if idx % 25 == 0:
          # Display current loss
          print(f'[Reconstruction] Epoch{e:4d}, Batch{idx:4d}, '
                f'Class Loss {cls_loss.item():0.5f}, Cont Loss '
                f'{cont_loss.item():0.5f}, '
                f'Loss {loss.item():0.5f}')
          # Record progress
          gcn_train_prog['rec_epoch'].append(e)
          gcn_train_prog['batch'].append(idx)
          gcn_train_prog['class_loss'].append(cls_loss.item())
          gcn_train_prog['cont_loss'].append(cont_loss.item())
          gcn_train_prog['loss'].append(loss.item())
    
    # Export GCN train logs
    io.write_csv(gcn_train_prog, self.csvdir, 'gcn.csv')
    io.plot_progress(gcn_train_prog, self.csvdir, 'gcn.png')

    del(loader)
    return

  def train_MMD(self):
    '''
    This method trains the full Meta-Sim architecture, it performs the GCN pretraining and then 
    performs the full training of the architecture.
    '''
    # GCN pre-training
    if self.opts['train_reconstruction']:
      if not self.opts['load_weights']:
        self.train_reconstruction()
    # Freeze encoder weights from the GCN encoder
    if self.opts['freeze_encoder']:
      self.model.freeze_encoder()

    # Get graph loader
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)

    # Initialize trainining progress dictionary
    mmd_train_prog = {'epoch':[], 'mmd_loss':[], 'dataset_gen_time':[], 'epoch_runtime':[]}
    # Define output data directory
    im_dir = os.path.join(self.imagedir, "most_recent")
    graph_dir = os.path.join(self.graphdir, "most_recent")
    fmap_dir = os.path.join(self.fmapsdir, "most_recent")
    io.makedirs(im_dir)
    io.makedirs(graph_dir)
    io.makedirs(fmap_dir)

    # For every epoch in max_epochs, specified in YAML experiment config file
    for e in tqdm(range(self.last_epoch, self.opts['max_epochs']+1), desc="Training Meta-Sim"):
      # Set seeds for epoch
      rn = utils.set_seeds(e)
      # Start timer
      estart = time.time()
      egen_start = time.time()
      # Disable gradient calculation since this is only inference
      with torch.no_grad():
        # Generate this epoch's data for task net
        i_g = 0
        # For every batch in the graph generator
        for idx, (g, x, m, adj) in tqdm(enumerate(loader), desc='Generating Data'):
          x, adj = x.float().to(self.device), adj.float().to(self.device)
          # Pass the sampled batch of graphs through GCN for reconstruction
          dec, dec_act = self.model(x, adj)
          f = dec_act.cpu().numpy() # Reconstructed features from GCN
          m = m.cpu().numpy() # m for mask
          # Update graph representation from generator
          g = self.generator.update(g, f, m) # We can feed this to the renderer quite easily to generate our own images
          #i_g = self.export_graph(g, graph_dir, i_g, gt=False)
          # Call Blender/Unity subprocess to generate batches of images from graphs
          r = self.generator.render(g)
          # Loop through all the elements in the graph
          for k in range(len(g)): 
            # Save most recent epoch distribution
            img, lbl = r[k]
            io.output_recent_data(im_dir, graph_dir, img, lbl, f[k], i_g, 'png')
            # Save key epoch distribution of images and graphs
            if e%self.opts['distprint_freq'] == 0:
              io.output_epoch_data(self.imagedir, self.graphdir, e, img, lbl, f[k], i_g, 'png')
            i_g+=1

      # Get epoch's data generation time
      epoch_gen_end = time.time()
      egen_time = epoch_gen_end - egen_start

      # Reset seeds to get exact same outputs
      rn2 = utils.set_seeds(e)
      for i in range(len(rn)):
        assert rn[i] == rn2[i], 'Random numbers generated are different'

      # Zero out gradients for first step
      self.optimizer.zero_grad()
      if self.opts['optim_mmd']['use_mmd_loss']:
        self.optimizer_mmd.zero_grad()
        # Create directory to store gradients
        inception_gradsdir = os.path.join(self.fmapsdir, 'gradients')
        io.makedirs_light(inception_gradsdir)
        egrads_inception = {"layer":[]}
      
      # Initialize epoch's gradient
      egrads = {"layer":[]}

      # Train distribution matching loss, for each batch of generated graphs
      for idx, (g, x, m, adj) in tqdm(enumerate(loader), desc="Training MMD"):
        # Get graph information from torch
        x, m, adj = (x.float().to(self.device), m.float().to(self.device),
          adj.float().to(self.device))
        
        # Pass sampled graphs to GCN for recunstruction, with sampling (what is sampling in this case?)
        dec, dec_act, log_probs = self.model(x, adj, m, sample=True)
        # Get real images from loader file
        im_real = torch.tensor(self.mmd_target_dataset.get_bunch_images(self.opts['num_real_images'])).to(self.device) # im.shape = (B, H, W, C)     
        # Get generated images and initialize backward pass through renderer
        im = self.renderer.render(g, dec_act, m)
        # Prepare input for Inception Network
        im_real = torch.div(im_real.permute(0,3,1,2), 255) # im.shape = (B, C, H, W)
        im = torch.div(im.permute(0,3,1,2), 255) # im.shape = (B, C, H, W)

        # Calculte Maximum Mean Discrepancy between generated and real image
        mmd, tgt_fmaps, gen_fmaps = self.mmd(im_real, im)
        mmd = mmd *self.opts['weight']['dist_mmd']
        # Export feature maps
        io.export_feature_maps(fmap_dir, tgt_fmaps, gen_fmaps, idx, e)
        if e%self.opts['distprint_freq']==0:
          efmaps_dir = os.path.join(self.fmapsdir, f'epoch_{e}')
          io.makedirs_light(efmaps_dir)
          io.export_feature_maps(efmaps_dir, tgt_fmaps, gen_fmaps, idx, e)

        # Backpropagate loss
        mmd.backward()
        self.optimizer.step()
        egrads = io.save_gradients(egrads, self.model.named_parameters(), idx)
        self.optimizer.zero_grad()
        if self.opts['optim_mmd']['use_mmd_loss']:
          self.optimizer_mmd.step()
          egrads_inception = io.save_gradients(egrads_inception, self.model.named_parameters(), idx)
          self.optimizer_mmd.zero_grad()
        
        # Print MMD loss
        if idx % self.opts['print_freq'] == 0:
          print(f'[Dist] Step: {idx} MMD: {mmd.item()}')

      # LR scheduler step
      self.lr_sched.step()
      if self.opts['optim_mmd']['use_mmd_loss']:
        self.lr_sched_mmd.step()

      # Save checkpoints and gradients from training
      if self.opts['optim_mmd']['use_mmd_loss']:
        io.save_epoch_weights(e, self.wdir, self.model, \
                            self.optimizer, self.mmd, self.optimizer_mmd)
        if e%2==0:
          io.write_csv(egrads_inception, inception_gradsdir, f'grads_e_{e}.csv')
      else:
        io.save_epoch_weights(e, self.wdir, self.model, \
                            self.optimizer, self.mmd)
        if e%2==0:
          io.write_csv(egrads, self.gradsdir, f'grads_e_{e}.csv')
        
      # Get epoch total time
      eend = time.time()
      etime = eend - estart
      # Store mmd performance logs and runtimes
      mmd_train_prog = io.save_mmd_progress(e, mmd, mmd_train_prog, egen_time, etime, self.csvdir)

  def generate_graphs(self):
    # Train Graph Convolutional Neural Network to reconstruct graphs
    if self.opts['train_reconstruction']:
      self.train_reconstruction()

    # Load graph dataset from grammar
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)

    with torch.no_grad():
      # Generate this epoch's data for task net
      i_g = 0
      # Generate the data
      # Generate graphs
      for idx, (g, x, m, adj) in tqdm(enumerate(loader), desc='Generating Graphs'):
        #i_gt = self.export_graph(g, graph_dir, i_gt, gt=True)
        x, adj = x.float().to(self.device), adj.float().to(self.device)
        # Generate graphs with GCN
        dec, dec_act = self.model(x, adj)
        f = dec_act.cpu().numpy() 
        m = m.cpu().numpy() 
        g = self.generator.update(g, f, m) 
        i_g = io.export_graph(g, self.graphdir, i_g, gt=False)      
        # Generate images using Blender as a renderer
        self.generator.render(g) # Render images with blender
        # Clean graph directory
        io.makedirs(self.graphdir)


# Run main Meta-Sim pipeline
if __name__ == '__main__':
  # Parse arguments in terminal command
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', required=True,
    type=str)
  opts = parser.parse_args()
  # Read yaml configuration
  opts = io.read_yaml(opts.exp)
  # Initialize Meta-Sim main trainer
  trainer = Trainer(opts)
  # Generate graphs and images with Blender (for the mooment)
  #trainer.generate_graphs()
  trainer.train_MMD()

