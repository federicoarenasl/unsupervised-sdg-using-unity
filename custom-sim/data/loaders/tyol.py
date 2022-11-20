"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import glob
import torch
import cv2
from torchvision.transforms.transforms import ToPILImage
import yaml
import os.path as osp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt 
import utils.io as io

class TYOLLoader(data.Dataset): # This is not yet implemented
  def __init__(self, root):
    self.root = root
    self.files = glob.glob(osp.join(self.root, '*.png'))
    self.files = [''.join(f.split('.')[:-1]) for f in self.files]
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def __getitem__(self, index):
    '''
    This is used for the target network training
    '''
    # Get image using OpenCV
    img = self.pull_image(index)
    # Get annotation
    target = self.pull_anno(index)
    
    # Transform image
    img = Image.fromarray(img) # convert to HWC
    # Resize it so our network can take any size
    data_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
    # Perform data transforms

    # There is a problem with the transformation, it changes the values
    # of the validation data
    img = data_transforms(img)

    # Inspect images
    io.inspect_img(img)

    #img = img / 255.0  # Normalize inputs

    return img, target

  def __len__(self):
    return len(self.files)

  def pull_image(self, index):
    data_id = self.files[index]
    img = cv2.imread(data_id + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

  def pull_image_norm(self, index):
    data_id = self.files[index]
    img = cv2.imread(data_id + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img).unsqueeze(0)

    return img

  def pull_anno(self, index):
    data_id = self.files[index]
    target = io.read_json(data_id + '.json')
    return target[0]['obj_class']

  def get_bunch_images(self, num):
    assert (num < len(self),
      'Asked for more images than size of data')
    
    idxs = np.random.choice(
      range(len(self)),
      num,
      replace=False,
    )

    # Get images and transform them to tensor
    bunch_images = np.array([self.pull_image(idx) for idx in idxs], dtype=np.float32)
    #bunch_images = [self.pull_image(idx) for idx in idxs]
    #bunch_images = torch.cat(bunch_images, dim=0).to(self.device)

    return bunch_images
  
  # Leave this as is, and add more functionalities if necessary