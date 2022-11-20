"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

from models.tasknet.mnist import MNISTModel
from models.tasknet.tyol_resnet18 import TYOLResnet # This will be implemented in the future with the TYOL data
#from models.tasknet.tyol_fasterrcnn import TYOLFrcnn # This will be implemented in the future with the TYOL data

def get_tasknet(name):
  if name == 'mnist':
    return MNISTModel
  if name == 'tyolBlender' or name == 'tyolUnity':
    return TYOLResnet
  else:
    raise NotImplementedError