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

class Trainer(object):
  def __init__(self, opts):
    self.opts = opts
    self.device = opts['device']
    # Logdir for generated images
    self.logdir = os.path.join(opts['logdir'],
      opts['imagedir'], opts['variant_name'])
    io.makedirs(self.logdir)

    # Logdir for generated csv files
    self.logdir_csv = os.path.join(opts['logdir'],
      opts['csvdir'], opts['variant_name'])
    io.makedirs(self.logdir_csv)

    # Set seeds
    rn = utils.set_seeds(opts['seed'])
    
    # Initialize Meta-Sim model and generator
    self.model = MetaSim(opts).to(self.device)
    self.generator = self.model.generator
    # Initialize Tasknet
    tasknet_class = get_tasknet(opts['dataset'])
    self.tasknet = tasknet_class(opts['task']).to(
      self.opts['task']['device'])

    # Initialize graph sampler
    sgl = get_scene_graph_loader(opts['dataset']) 
    self.scene_graph_dataset = sgl(self.generator, self.opts['epoch_length'])

    # Rendering layer
    self.renderer = RenderLayer(self.generator, self.device)

    # Initialize Maximum Mean Discrepancy calculator
    self.mmd = MMDInception(device=self.device, 
      resize_input=self.opts['mmd_resize_input'], 
      include_image=False, dims=self.opts['mmd_dims'])
    # Get target dataset loader
    dl = get_loader(opts['dataset'])
    self.target_dataset = dl(self.opts['task']['val_root'])

    # Initialize optimizer
    self.optimizer = torch.optim.Adam(
      self.model.parameters(),
      lr = opts['optim']['lr'],
      weight_decay = opts['optim']['weight_decay']
    )

    # Initialize Learning Rate scehduler
    self.lr_sched = torch.optim.lr_scheduler.StepLR(
      self.optimizer,
      step_size = opts['optim']['lr_decay'],
      gamma = opts['optim']['lr_decay_gamma']
    )

  def train_reconstruction(self):
    # Load graph dataset from graph sampler
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)
    # Initalize dictionary to store logs
    gcn_train_prog = {'rec_epoch':[], 'batch':[], 'class_loss':[], 'cont_loss':[], 'loss':[]}
    # For every epoch in reconstruction_epochs
    for e in range(self.opts['reconstruction_epochs']):
      # For every batch in the graph loader
      for idx, (g, x, m, adj) in enumerate(loader):
        # g: scene graph, x: encoded features, m : mutability mask, adj: adjacency matrix
        x, m, adj = x.float().to(self.device), m.float().to(self.device),\
          adj.float().to(self.device)
        # Decoded graphs, after GCN encoder-decoder
        dec, dec_act = self.model(x, adj) 
        # Classification on classes in features
        cls_log_prob = F.log_softmax(dec[..., :self.model.num_classes], dim=-1)
        # Negative log likelihood loss
        cls_loss = -torch.mean(torch.sum(
          cls_log_prob * x[..., :self.model.num_classes], 
          dim=-1))
        # Weight loss
        cls_loss *= self.opts['weight']['class'] # Class loss
        # Reconstruction loss for numerical features
        cont_loss = F.mse_loss(dec_act[..., self.model.num_classes:], 
          x[..., self.model.num_classes:]) # Reconstruction loss (Mean Squared Error) distance between both reconstructions
        # Add up both losses
        loss = cls_loss + cont_loss

        if idx % 25 == 0: 
          print(f'[Reconstruction] Epoch{e:4d}, Batch{idx:4d}, '
                f'Class Loss {cls_loss.item():0.5f}, Cont Loss '
                f'{cont_loss.item():0.5f}, '
                f'Loss {loss.item():0.5f}')
          
          gcn_train_prog['rec_epoch'].append(e)
          gcn_train_prog['batch'].append(idx)
          gcn_train_prog['class_loss'].append(cls_loss.item())
          gcn_train_prog['cont_loss'].append(cont_loss.item())
          gcn_train_prog['loss'].append(loss.item())

        # Update GCN parameters
        self.optimizer.zero_grad()
        # Backpropagate loss
        loss.backward()
        # Increase one step on Adam optimizer
        self.optimizer.step()
    
    # Export GCN train logs
    io.write_csv(gcn_train_prog, self.logdir_csv, 'gcn.csv')


    del(loader)
    return

  def train(self):
    # Pre-train Graph Convolutional Network
    if self.opts['train_reconstruction']:
      self.train_reconstruction()
    # Freeze gradient parameters from GCN encoder
    if self.opts['freeze_encoder']:
      self.model.freeze_encoder()
    # Get graph loader 
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)
    # Baseline for moving average
    baseline = 0.
    alpha = self.opts['moving_avg_alpha']
    # Init train progress dictionary
    mmd_train_prog = {'epoch':[], 'idx':[], 'mmd_loss':[]}
    task_train_prog = {'epoch':[], 'idx':[], 'mmd_loss':[], 'task_loss':[], 'loss':[], 'task_acc':[]}

    # For every epoch in max_epochs, start Meta-Sim training
    for e in range(self.opts['max_epochs']):
      # Set seeds for epoch
      rn = utils.set_seeds(e)

      # Disable gradient calculation since this is only inference
      with torch.no_grad():
        # Generate this epoch's data for task net
        i = 0
        # Define output directory
        out_dir = self.logdir
        # Loop through batches of graphs in loader
        for idx, (g, x, m, adj) in tqdm(enumerate(loader), desc='Generating Data'):
          # Revert from torch tensor to numpy
          x, adj = x.float().to(self.device), adj.float().to(self.device)
          # Generate graph with GCN
          dec, dec_act = self.model(x, adj)
          # Reconstructed features from GCN
          f = dec_act.cpu().numpy() 
          # Get mutability mast
          m = m.cpu().numpy()
          # Update the graph representation
          g = self.generator.update(g, f, m) 
          # Feed generated graph to renderer
          r = self.generator.render(g)
          # Loop through every graph in the batch to generate label and image
          for k in range(len(g)): 
            img, lbl = r[k]
            out_img = os.path.join(out_dir, f'{str(i).zfill(6)}.jpg')        
            out_lbl = os.path.join(out_dir, f'{str(i).zfill(6)}.json')
            io.write_img(img, out_img)
            io.write_json(lbl, out_lbl)
            i+=1

      # Train tasknet on generated images
      acc = self.tasknet.train_from_dir(out_dir)
      # Compute moving average for REINFORCE policy gradient
      if e > 0:
        baseline = alpha * acc + (1-alpha) * baseline
      else:
        # initialize baseline to acc
        baseline = acc
      # Reset seeds to get exact same outputs
      rn2 = utils.set_seeds(e)
      for i in range(len(rn)):
        assert rn[i] == rn2[i], 'Random numbers generated are different'

      # Zero out gradients for first step
      self.optimizer.zero_grad()
      # Train distribution matching metric and task loss
      for idx, (g, x, m, adj) in enumerate(loader):
        x, m, adj = (x.float().to(self.device), m.float().to(self.device),
          adj.float().to(self.device))

        dec, dec_act, log_probs = self.model(x, adj, m, sample=True)
        # sample here
        
        # Get real test images
        im_real = torch.from_numpy(self.target_dataset.get_bunch_images(
          self.opts['num_real_images'])).to(self.device)

        # Get generated synthetic images
        im = self.renderer.render(g, dec_act, m)

        if self.opts['dataset'] == 'mnist':
          # add channel dimension and repeat 3 times for MNIST
          im = im.unsqueeze(1).repeat(1,3,1,1) / 255.
          im_real = im_real.permute(0,3,1,2).repeat(1,3,1,1) / 255.

        # Calculate Maximum Mean Discrepancy between real and generated image
        mmd = self.mmd(im_real, im) * self.opts['weight']['dist_mmd']
        # If use_task_loss, perform REINFORCE policy gradient losss calculation
        if self.opts['use_task_loss']:
          task_loss = -1 * torch.mean((acc - baseline) * log_probs)
          # Add both losses
          loss = mmd + task_loss
          # Backpropagate loss to GCN
          loss.backward()
        else:
          # Backpropagate only MMD loss
          mmd.backward()
          # Perform Adam optimization step
          self.optimizer.step()
          # Zero out gradients
          self.optimizer.zero_grad()

        if idx % self.opts['print_freq'] == 0:
          print(f'[Dist] Step: {idx} MMD: {mmd.item()}')
          if self.opts['use_task_loss']:
            print(f'[Task] Reward: {acc}, Baseline: {baseline}')
            # Store task performance logs
            task_train_prog['epoch'].append(e)
            task_train_prog['idx'].append(idx)
            task_train_prog['mmd_loss'].append(mmd.item())
            task_train_prog['task_loss'].append(task_loss.item())
            task_train_prog['loss'].append(loss.item())
            task_train_prog['task_acc'].append(acc)
          # debug information
          print(f'[Feat] Step: {idx} {dec_act[0, 2, 15:].tolist()} {x[0, 2, 15:].tolist()}')
          # To debug, this index is the loc_x, loc_y, yaw of the 
          # digit in MNIST
          # Store mmd performance logs
          mmd_train_prog['epoch'].append(e)
          mmd_train_prog['idx'].append(idx)
          mmd_train_prog['mmd_loss'].append(mmd.item())
          
      if self.opts['use_task_loss']:
        # Perform Adam optimization step 
        self.optimizer.step()
        # Zero out gradients
        self.optimizer.zero_grad()

      # LR scheduler step
      self.lr_sched.step()
    
    # Output train logs
    io.write_csv(mmd_train_prog, self.logdir_csv, 'mmd.csv')
    io.write_csv(task_train_prog, self.logdir_csv, 'task.csv')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', required=True,
    type=str)
  opts = parser.parse_args()
  opts = io.read_yaml(opts.exp)

  trainer = Trainer(opts)
  trainer.train()
