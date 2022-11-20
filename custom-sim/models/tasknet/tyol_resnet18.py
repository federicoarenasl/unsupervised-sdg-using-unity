"""
Adapted from:
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet
from data.loaders import get_loader
from torchvision import datasets, models, transforms
import time
import copy
from tqdm import tqdm
import utils.io as io
import matplotlib.pyplot as plt

class TYOLResnet(nn.Module):
  def __init__(self, opts):
    super(TYOLResnet, self).__init__()
    # Load experiment configuration
    self.opts = opts
    self.device = opts['device']
    # Initialize the ResNet18 model
    self.resnet = self.initialize_model(num_classes=2)
    # Send the model to GPU
    self.resnet = self.resnet.to(self.device)
    # Initialize optimizer
    self.optimizer = optim.SGD(self.resnet.parameters(), lr=0.01, momentum=0.5)
    # Setup the loss function
    self.loss =  nn.CrossEntropyLoss()
    # Get validation loader
    self.dataset_class = get_loader(opts['dataset']) # The data loading is already good
    self.val_dataset = self.dataset_class(opts['val_root'])
    self.val_loader = torch.utils.data.DataLoader(self.val_dataset, 
      opts['batch_size'], shuffle=True, num_workers=0)

    if 'reload' in opts.keys():
        self.reload(opts['reload'])

  def initialize_model(self, num_classes):
    '''
    Receives the output number of classes, feature_extract, and the use_pretrained flag
    and outputs the initialized model.
    '''
    # Initialize input size and model variable
    resnet = None
    # Initialize model
    resnet = models.resnet18(pretrained = True)
    self.set_parameter_requires_grad(resnet, True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    return resnet
  
  def set_parameter_requires_grad(self, model, feature_extracting):
    '''
    Receives the model and a the boolean feature_extracting variable, which if 
    set to True, uses the pretrained weights from ImageNet and updates the parameters of
    the model accordingly.
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

  def train_from_dir(self, root):
    '''
    Takes a pretrained model, the dataloaders, the momentum criterium, the optimizer, the torch device, the number
    of epochs and outputs the trained model with the progress information.
    '''
    # Start counting the training time
    since = time.time()

    # Initialize variables that will store the training information
    train_acc_history = []
    train_loss_history = []

    # Get the best model weights from the inputted model
    best_model_wts = copy.deepcopy(self.resnet.state_dict())
    best_acc = 0.0

    # Set model to training mode
    self.resnet.train()  
    
    # Get generated dataset
    dataset = self.dataset_class(root)
    data_loader = torch.utils.data.DataLoader(dataset,
      self.opts['batch_size'], shuffle=True, num_workers=0) # What if we don't shuffle this

    # Loop through epochs to start training
    for epoch in tqdm(range(self.opts['epochs']), desc='Training ResNet'):
      print('Epoch {}/{}'.format(epoch, self.opts['epochs'] -1))
      print('-'*10)
      # Initialize running losses
      running_loss = 0.0
      running_corrects = 0
      # Iterate over data.
      print(f"Looking at data in {root}")
      for idx, (inputs, labels) in enumerate(data_loader):
        # Send input and labels to gpu/cpu
        # Inspect images being fed to model
        #io.inspect_img(inputs[0])

        inputs = inputs.to(self.device)


        labels = labels.to(self.device)
        # Zero out the parameter gradients
        self.optimizer.zero_grad()
        # Get model outputs and calculate loss
        outputs = self.resnet(inputs)
        loss = self.loss(outputs, labels)
        _, preds = torch.max(outputs, 1)
        # Backpropagate parameter modifications
        loss.backward()
        self.optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      # Correct epoch losses
      epoch_loss = running_loss / len(data_loader.dataset)
      epoch_acc = running_corrects.double() / len(data_loader.dataset)
      # Print loss
      print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
      # Store values from training
      train_acc_history.append(epoch_acc.numpy())
      train_loss_history.append(epoch_loss)

    # Calculate total training time
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    self.resnet.load_state_dict(best_model_wts)

    # Call destructors
    del(dataset, data_loader)

    # Validate on real data
    acc = self.test()
    
    return acc

  def test(self):
    self.resnet.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for img, target in self.val_loader:
        img, target = img.to(self.device), target.to(self.device)

        # Inspect images being fed to model
        io.inspect_img(img[0])

        out = self.resnet(img)
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(self.val_dataset)
    print('###############################')
    print(f'[TaskNet] Accuracy: {acc:3.2f}')
    print('###############################')
    
    return acc

  def save(self, fname, acc):
    print(f'Saving task network at {fname}')
    save_state = {
        'state_dict': self.state_dict(),
        'acc': acc
    }

    torch.save(save_state, fname)

  def reload(self, fname):
    print(f'Reloading task network from {fname}')
    state_dict = torch.load(fname, map_location=lambda storage, loc: storage)
    self.load_state_dict(state_dict['state_dict'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', required=True, type=str)
  opts = parser.parse_args()
  opts = io.read_yaml(opts.exp)

  tyol_tasknet = TYOLResnet(opts['task'])
  root = '/home/federicoarenasl/Documents/Federico/UoE/MSC_AI/Thesis_project/implementation/meta-sim/custom-sim/logs/images/tyol/exp_2'
  tyol_tasknet.train_from_dir(root)

