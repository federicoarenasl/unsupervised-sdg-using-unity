"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import time

class RenderTorch(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graphs, features, masks, generator, device):
    if device=='cpu': # Include cpu variation for fast prototyping
      features = features.cpu().detach().numpy()
    else:
      features = features.cpu().numpy()
    masks = masks.cpu().numpy()

    # update in place
    graphs = generator.update(graphs, features, masks)
    images = [r[0] for r in generator.render(graphs)]
    images = torch.tensor(images, dtype=torch.float32, device=device)
    
    # Here, in due time, it would be important to implement a preprocessing
    # step to resize the generated images to a common size, the same size as the 
    # real datasset

    # Save in ctx for backward
    ctx.graphs = graphs
    ctx.features = features
    ctx.masks = masks
    ctx.generator = generator
    ctx.device = device

    return images

  @staticmethod
  def backward(ctx, output_grad):
    '''
    This static method is still quite obscure. The baseline implementation does not
    call it.
    '''
    graphs = ctx.graphs
    features = ctx.features
    masks = ctx.masks
    generator = ctx.generator
    device = ctx.device

    out_grad = torch.zeros(features.shape, dtype=torch.float32,
      device=device)

    delta = 0.005 #previously 0.03

    for b in range(features.shape[0]):
      idxs = np.array(np.nonzero(masks[b])).transpose()
    
      for idx in idxs:
        #time.sleep(0.1)
        # f(x+delta)
        features[b, idx[0], idx[1]] += delta
        tmp_graph = generator.update(graphs[b], features[b],
          masks[b])
        img_plus = generator.render(tmp_graph)[0][0] / 255.0
        # f(x-delta)
        features[b, idx[0], idx[1]] -= delta #previously 2*delta
        tmp_graph = generator.update(graphs[b], features[b],
          masks[b])
        img_minus = generator.render(tmp_graph)[0][0] / 255.0

        grad = ((img_plus - img_minus) / 2*delta).astype(np.float32)
        grad = torch.from_numpy(grad).to(device)

        out_grad[b, idx[0], idx[1]] = (output_grad * grad).sum()

        # back to normal
        features[b, idx[0], idx[1]] += delta
        tmp_graph = generator.update(graphs[b], features[b], masks[b])

    return None, out_grad, None, None, None
