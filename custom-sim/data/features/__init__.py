"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

from data.features.mnist import MNISTFeatures
from data.features.tyol import TYOLFeatures

def get_features(name):
  if name == 'mnist':
    return MNISTFeatures
  if name == 'tyol':
    return TYOLFeatures
  else:
    raise NotImplementedError