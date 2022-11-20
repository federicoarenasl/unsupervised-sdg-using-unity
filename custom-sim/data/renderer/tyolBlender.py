"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json
import glob
import struct
import os
import os.path as osp
import numpy as np
from PIL import Image
import subprocess
import utils.io as io
import cv2


class TYOLBlenderRenderer(object):
    """
  Domain specific renderer definition
  Main purpose is to be able to take a scene
  graph and return its corresponding rendered image
  """
    def __init__(self, config):
        self.config = config
        self.graphdir = "logs/graphs/tyol/tyol_blender"
        self.temp_datagen = "logs/images/tyol/temp"
        io.makedirs(self.temp_datagen)
        self._init_data()

    def _init_data(self):
        asset_dir = self.config['attributes']['asset_dir']
        self.data = {}

        with open(osp.join(asset_dir, 'labels'), 'rb') as flbl:
            _, size = struct.unpack('>II', flbl.read(8))
            lbls = np.fromfile(flbl, dtype=np.int8)

        with open(osp.join(asset_dir, 'images'), 'rb') as fimg:
            _, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
            imgs = np.fromfile(fimg,
                               dtype=np.uint8).reshape(len(lbls), rows, cols)

        for lbl in np.unique(lbls):
            idxs = (lbls == lbl)
            self.data[lbl] = imgs[idxs]

        return

    def render(self, graphs):
        """
    Render a batch of graphs to their 
    corresponding images using a Blender subprocess
    """
        # Refresh temporary directory
        io.makedirs(self.temp_datagen)

        if not isinstance(graphs, list):
            graphs = [graphs]
        # Generate graphs
        self.export_graph(graphs, self.graphdir)

        # Here is where we call the renderer subprocess
        blender_source_path = self.config['attributes']['blender_path']
        blender_file_path = self.config['attributes']['blender_file_path']
        blender_script_path = self.config['attributes']['blender_script_path']
        target_image_dir = self.temp_datagen

        # Display is for running headless
        cmd = [
            blender_source_path, "-b", "-P", blender_script_path,"--",
            self.graphdir, blender_file_path, target_image_dir
        ]

        # Start rendering batch
        subprocess.run(cmd, capture_output=False)

        # Pull generated images from temporary location and return them
        rendered = self.parse_temp()

        return rendered

    def parse_temp(self):
        # Get glob of files
        self.files = glob.glob(osp.join(self.temp_datagen, '*.png'))
        # Get names of files
        self.files = [''.join(f.split('.')[:-1]) for f in self.files]
        rendered = []
        # Gather images and annotations
        for i in range(len(self.files)):
            img = self.pull_image(i)
            target = self.pull_anno(i)
            rendered.append((img, target))
        # Return list of tuples with image and annotation pairs
        return rendered

    def pull_image(self, index):
        data_id = self.files[index]
        img = cv2.imread(data_id + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape
        assert channels == 3

        return img

    def pull_anno(self, index):
        data_id = self.files[index]
        target = io.read_json(data_id + '.json')
        return target

    def export_graph(self, graphs, graph_dir):
        # Loop through every graph in the batch
        i = 0
        for k in range(len(graphs)):
            #gout_graph = os.path.join(graph_dir, f'ggraph_{str(i).zfill(6)}.jpg')
            gout_graph_json = os.path.join(graph_dir,
                                           f'ggraph_{str(i).zfill(6)}.json')
            # Save graph as plot and as json, comment plot_graph for mass data generation and increased speed
            #io.plot_graph(G[k], gout_graph)
            io.write_graph_json(graphs[k], gout_graph_json)
            i += 1


if __name__ == '__main__':
    config = json.load(open('data/generator/config/mnist.json', 'r'))
    re = Renderer(config)
