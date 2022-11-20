"""
Parts of code licensed under Apache 2.0 License from https://github.com/napsternxg/pytorch-practice/

Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.
"""
import torch
import torch.nn as nn
from torchvision.transforms.transforms import ToPILImage, ToTensor
from models.layers.mmd.inception_v3 import InceptionV3
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from geomloss import SamplesLoss
import numpy as np

class MMDGeomInception(nn.Module):
  def __init__(self, device='cuda', resize_input=True, feature_extract=True, out_features=50, loss_type="gaussian"):
    super(MMDGeomInception, self).__init__()
    print(f"Initializing MMDGeomInception on device {device}, and out_features{out_features}")
    # Get device
    self.device = device
    # Whether to resize input or not
    self.resize_input = resize_input
    # Initialize Inception network
    self.model = models.inception_v3(pretrained=True, transform_input=True).to(device)
    # Set parameters to false
    if feature_extract:
      for param in self.model.parameters():
          param.requires_grad = False
    # Handle the auxilary net
    self.model.AuxLogits.fc = nn.Linear(768, out_features)
    # Handle the primary net
    self.model.fc = nn.Linear(2048,out_features)
    # Initialize Gomloss losses
    self.loss_mmd = SamplesLoss(loss=loss_type, blur=0.001) # Ask Patric about this, what should this blur be set to?

  def get_features(self, images):
    if self.resize_input:
      images = F.interpolate(images, size=(299, 299), mode='bilinear')

    images = self.normalize_images(images)
    self.model.cuda()
    out_feats, aux_out_feats = self.model(images)

    return out_feats, aux_out_feats
  
  def normalize_images(self, images):
    # Input: tensor of size (B, C, H, W)
    normalized_images = []
    for image in images:
      preprocess = transforms.Compose([
                                      transforms.ToPILImage(),
                                      transforms.Resize((299,299)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
      
      # Prepare tensor to PIL conversion
      image = image.permute(1,2,0)
      image = np.uint8(image.detach().cpu().numpy())
      # Normalize images
      normalized_image = preprocess(image).to(self.device)
      # Prepare for batch concatenation
      normalized_image = torch.tensor(normalized_image).unsqueeze(0)
      normalized_images.append(normalized_image)

    # Concatenate batch of normalized images
    images = torch.cat(normalized_images, dim=0)
    
    return images

  def forward(self, real, gen, is_feat=False):
    if is_feat:
      # If we cache features on a huge set and
      # want mmd on the bigger set
      x = real
      y = gen
    else:
      with torch.no_grad():
        x, x_aux = self.get_features(real)
      y, y_aux = self.get_features(gen)

    loss1 = self.loss_mmd(x, y)
    loss2 = self.loss_mmd(x_aux, y_aux) 
    mmd_loss = loss1 + 0.4*loss2
    #return loss

    return mmd_loss, [x, x_aux], [y, y_aux]