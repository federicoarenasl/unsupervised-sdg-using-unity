"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

from data.generator.mnist import MNISTGenerator
from data.generator.tyolBlender import TYOLBlenderGenerator
from data.generator.tyolUnity import TYOLUnityGenerator

def get_generator(name):
  if name == 'mnist':
    return MNISTGenerator
  if name == 'tyolBlender':
    return TYOLBlenderGenerator
  if name == 'tyolUnity':
    return TYOLUnityGenerator
  else:
    raise NotImplementedError