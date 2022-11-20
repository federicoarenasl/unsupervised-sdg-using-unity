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
import time

class TYOLUnityRenderer(object):
    """
  Domain specific renderer definition
  Main purpose is to be able to take a scene
  graph and return its corresponding rendered image
  """
    def __init__(self, config):
        self.config = config
        self.build_path = self.config['attributes']['build_path']
        self.stream_path = os.path.join(self.build_path, "StreamingAssets")
        self.graphdir = os.path.join(self.stream_path, "Graphs")
        self.temp_datagen = os.path.join(self.stream_path, "Snapshots")
        self.done_dir = os.path.join(self.stream_path, "Done")
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
        if not isinstance(graphs, list):
            graphs = [graphs]
        
        # Generate graphs
        batch_size = len(graphs)
        # The 4 lines below can be deleted
        if batch_size == 1:
            #time.sleep(0.1)
            graphs = [graphs[0]]*1
            batch_size = len(graphs)

        # Export graphs to folder
        self.export_graph(graphs, self.graphdir, batch_size)

        # Pull generated images from temporary location and return them
        rendered = self.parse_temp(batch_size)
        
        # Refresh directories
        io.makedirs(self.temp_datagen)
        io.makedirs(self.done_dir)

        return rendered

    def parse_temp(self, batch_size):
        # Check if full batch is in target directory
        #print("Checking DONE in Snapshots folder")
        not_batch = True
        temp_count_2 = 0
        breaker_counter2 = 0 
        while not_batch:
            temp_count_2 +=1 
            # Get glob of files
            self.files = glob.glob(osp.join(self.temp_datagen, '*.png'))
            self.snap_txt = glob.glob(osp.join(self.temp_datagen, '*.txt'))
            self.graphs = glob.glob(osp.join(self.graphdir, '*.txt'))
            # Get names of files
            self.files = [''.join(f.split('.')[:-1]) for f in self.files]
            rendered = []
            if len(self.graphs) == 0:
                if len(self.snap_txt) == 1:
                    snap_size = os.stat(self.snap_txt[0]).st_size
                    #print(f"### SNAP file size: {snap_size}  ###")
                    if snap_size == 4:
                        if len(self.files) == batch_size:
                            with open(self.snap_txt[0], "r") as my_snap:
                                snap = my_snap.read()
                                if snap == 'Done':
                                        not_batch = False
            else:
                if temp_count_2%1000 == 1:
                    breaker_counter2+=1
                    if breaker_counter2 > 5:
                        print("\n\nStuck in SNAP loop")
                        print(f"WARNING NO TEXT FILE: {self.snap_txt}")
                        print("BREAKING OUT OF LOOP")
                        exit()
                not_batch = True
                
        # Check if batch generation is done
        not_done = True
        temp_count = 0
        breaker_counter = 0
        while not_done:
            temp_count += 1
            self.txt_file = glob.glob(osp.join(self.done_dir, '*.txt'))
            if len(self.txt_file) == 1:
                file_size = os.stat(self.txt_file[0]).st_size
                #print(f"### DONE file size: {os.stat(self.txt_file[0]).st_size}  ###")
                if file_size == 4:
                    with open(self.txt_file[0], 'r') as my_txt:
                        text = my_txt.read()
                        if text == 'Done':
                            not_done = False
            else:
                if temp_count%1000==0:
                    breaker_counter+=1
                    if breaker_counter > 5:
                        print("\n\nStuck in DONE loop")
                        print(f"WARNING NO TEXT FILE: {self.txt_file}")
                        print("BREAKING OUT OF LOOP")
                        exit()
                not_done = True

        # Gather images and annotations
        for i in range(len(self.files)):
            #time.sleep(0.00001)
            img = self.pull_image_robust(i)
            target = self.pull_anno_robust(i)
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
    
    def pull_image_robust(self, index):
        not_image = True
        while not_image:
            #print("Going into loop")
            try:
                data_id = self.files[index]
                img = cv2.imread(data_id + '.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                height, width, channels = img.shape
                assert channels == 3

                return img
            
            except:
                print(f"FILE {index} NOT READY")
                not_image = False

    def pull_anno(self, index):
        data_id = self.files[index]
        target = io.read_json(data_id + '.json')
        return target
    
    def pull_anno_robust(self, index):
        not_file = True
        while not_file:
            try:
                data_id = self.files[index]
                target = io.read_json(data_id + '.json')
                return target
            except:
                print(f"FILE {index} NOT READY")
                not_file = True

    def export_graph(self, graphs, graph_dir, batch_size):
        # Loop through every graph in the batch
        i = 0
        for k in range(len(graphs)):
            #gout_graph = os.path.join(graph_dir, f'ggraph_{str(i).zfill(6)}.jpg')
            gout_graph_json = os.path.join(graph_dir,
                                           f'ggraph_{str(i).zfill(6)}_{batch_size}.json')
            # Save graph as plot and as json, comment plot_graph for mass data generation and increased speed
            #io.plot_graph(G[k], gout_graph)
            io.write_graph_json(graphs[k], gout_graph_json)
            i += 1


if __name__ == '__main__':
    config = json.load(open('data/generator/config/mnist.json', 'r'))
    re = Renderer(config)
