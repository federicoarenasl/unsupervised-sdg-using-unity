import bpy
import json
import sys
import os
from pathlib import Path
import math
import numpy as np

class TYOLTarget():
    def __init__(self, scene_file, target_dir):
        '''
        graph_dir: directory where all graphs generated by the GCN are stored
        scene_file: file where the Blender scene file is stored
        target_dir: target directory where the generated images are being stored
        '''
        # Initialize directory variables
        self.target_dir = target_dir
        # Input number of classes in the scene
        self.obj_classes = ['1'] # Add more if necessary
        # Define main elements inherent to a Blender scene
        self.scene_essentials = ['Light', 'Camera', 'Background', 'Axis', 'Surface']
        # Load all Blender objects
        self.scene_nodes = self.load_scene(scene_file)
        # Define global camera and scene parameters
        self.camera = self.scene_nodes['Camera']
        self.scene = bpy.data.scenes['Scene']
        # Retrieve tgt resolution
        self.resolution_x = self.scene.render.resolution_x
        self.resolution_y = self.scene.render.resolution_y
        # Init config generator
        self.num_samples = 200
        # Define attributes to modify from scene and proportion [tgt_val1, tgt_val2, proportion]
        self.target_attr = {'1':{'yaw':[0, 90, 0.2]}}
        print("Init done...")
        
    def load_scene(self, scene_file):
        '''
        This method takes as input the file where the scene is located and returns
        all the bpy.data.objects in the scene.
        '''
        # Load scene
        bpy.ops.wm.open_mainfile(filepath=scene_file)
        objects = {}
        # Loop through all objects in the scene
        for obj in bpy.data.objects:
            # Add all inserted assets
            if obj.name in self.obj_classes:
                objects[obj.name] = obj
            # Add al scene essential elements
            if obj.name in self.scene_essentials:
                objects[obj.name] = obj
        # Return dictionary containing the Blender objects
        return objects

    def render(self, i):
        '''
        This method takes the filename as input and renders the image of the current
        scene to this location.
        '''
        # Load image path
        img_file = os.path.join(self.target_dir,  f'tgt_{str(i).zfill(6)}.png')
        img_path = Path(img_file)
        # Render image in Blender
        print("Rendering image...")
        bpy.context.scene.render.filepath = str(img_path)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

    def generate_tgt(self):
        tgt_proportion = self.target_attr['1']['yaw'][-1]
        tgt_value1 = self.target_attr['1']['yaw'][0]
        tgt_value2 = self.target_attr['1']['yaw'][1]
        for samples in range(self.num_samples):
            if samples < self.num_samples*tgt_proportion:
                # Generate images with target attribute
                self.modify_scene_attr(self.scene_nodes['1'], 'yaw', tgt_value1)
                # Render images
                self.render(samples)
                # Return label
                self.output_label(self.scene_nodes['1'], '1', samples, tgt_value1)

            else:
                # Generate images with target attribute
                self.modify_scene_attr(self.scene_nodes['1'], 'yaw', tgt_value2)
                # Render images
                self.render(samples)
                # Return label
                self.output_label(self.scene_nodes['1'], '1', samples, tgt_value2)

    def modify_scene_attr(self, scene_obj, mutable_attr, value):
        '''
        This method takes the following arguments
            scene_obj: Blender scene object
            mutable_attr: mutable attribute to change in scene
            value: value of the mutable attribute
            node_class: class of the node in graph 
        And modifies the values of the scene and saves the scene
        '''

        if mutable_attr == 'yaw':
            # Modify scene's value
            scene_obj.rotation_euler[2] = math.radians(value)
            # Save scene
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

        if mutable_attr == 'pitch':
            # Modify scene's value
            scene_obj.rotation_euler[1] = math.radians(value)
            # Save scene
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

        if mutable_attr == 'roll':
            # Modify scene's value
            scene_obj.rotation_euler[0] = math.radians(value)
            # Save scene
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

        if mutable_attr == 'loc_x':
            # Modify scene's value
            scene_obj.location[0] = value
            # Save scene
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        
        if mutable_attr == 'loc_y':
            # Modify scene's value
            scene_obj.location[1] = value
            # Save scene
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    
    def output_label(self, obj, object_class, i, tgt_value):
        '''
        This function takes no input and outputs the complete string with the coordinates
        of all the objects in view in the current image
        '''
        main_text_coordinates = '' # Initialize the variable where we'll store the coordinates
        print("     On object:", obj)
        b_box = self.find_bounding_box(obj) # Get current object's coordinates
        if b_box: # If find_bounding_box() doesn't return None
            print("         Initial coordinates:", b_box)
            json_coordinates = self.format_coordinates(b_box, tgt_value, object_class) # Reformat coordinates to YOLOv3 format
            print("         YOLO-friendly coordinates:", json_coordinates)
            #main_text_coordinates = main_text_coordinates + text_coordinates # Update main_text_coordinates variables whith each, multi-class case
                                                                                # line corresponding to each class in the frame of the current image
        else:
            print("         Object not visible")
            pass
        
        with open(os.path.join(self.target_dir,  f'tgt_{str(i).zfill(6)}.json'), 'w') as json_file:
            json_file.write(json.dumps([json_coordinates]))

        #return main_text_coordinates # Return all coordinates

    def format_coordinates(self, coordinates, tgt_value, classe):
        '''
        This function takes as inputs the coordinates created by the find_bounding box() function, the current class,
        the image width and the image height and outputs the coordinates of the bounding box of the current class
        '''
        # If the current class is in view of the camera
        if coordinates: 
            ## Change coordinates reference frame
            x1 = (coordinates[0][0])
            x2 = (coordinates[1][0])
            y1 = (1 - coordinates[1][1])
            y2 = (1 - coordinates[0][1])

            ## Get final bounding box information
            width = (x2-x1)  # Calculate the absolute width of the bounding box
            height = (y2-y1) # Calculate the absolute height of the bounding box
            # Calculate the absolute center of the bounding box
            cx = (x1 + (width/2))
            cy = (y1 + (height/2))

            print(f"Resolution x: {self.resolution_x}")
            print(f"Resolution y: {self.resolution_y}")

            ## Formulate line corresponding to the bounding box of one class
            width = width*self.resolution_x
            height = height*self.resolution_y
            cx = cx*self.resolution_x
            cy = cy*self.resolution_y
            
            json_coordinates = {"obj_class":int(classe),"yaw": tgt_value,"bbox":[cx, cy, width, height]}
            
            return json_coordinates

        # If the current class isn't in view of the camera, then pass
        else:
            pass

    def find_bounding_box(self, obj):
        """
        Returns camera space bounding box of the mesh object.
        Gets the camera frame bounding box, which by default is returned without any transformations applied.
        Create a new mesh object based on self.carre_bleu and undo any transformations so that it is in the same space as the
        camera frame. Find the min/max vertex coordinates of the mesh visible in the frame, or None if the mesh is not in view.
        :param scene:
        :param camera_object:
        :param mesh_object:
        :return:
        """

        """ Get the inverse transformation matrix. """
        matrix = self.camera.matrix_world.normalized().inverted()
        """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
        mesh = obj.to_mesh(preserve_all_data_layers=True)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)

        """ Get the world coordinates for the camera frame bounding box, before any transformations. """
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                """ Perspective division """
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)


        """ Image is not in view if all the mesh verts were ignored """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Image is not in view if both bounding points exist on the same side """
        if min_x == max_x or min_y == max_y:
            return None

        """ Figure out the rendered image size """
        render = self.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac
        
        ## Verify there's no coordinates equal to zero
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)

# Run the main code
if __name__ == "__main__":
    # If we are running script from Blender
    if "--" not in sys.argv:
        print("RUNNING IN BLENDER")
        # Set fixed values for class' arguments
        scene_file = "/home/federicoarenasl/Documents/Federico/UoE/MSC_AI/Thesis_project/implementation/data/single_object_dataset/v1_2_class/models/scene_incremental.blend"
        graph_dir = "/home/federicoarenasl/Documents/Federico/UoE/MSC_AI/Thesis_project/implementation/meta-sim/custom-sim/logs/graphs/tyol/exp_2"
        target_image_dir = "/home/federicoarenasl/Documents/Federico/UoE/MSC_AI/Thesis_project/implementation/meta-sim/custom-sim/logs/images/tyol/exp_2"
    
    # Else we are running this script as a subprocess
    else:
        print("RUNNING IN BG MODE")
        # Grab first argument after -- and so on
        scene_idx = sys.argv.index("--") + 2
        target_image_idx = sys.argv.index("--") + 3
        scene_file = sys.argv[scene_idx]
        target_image_dir = sys.argv[target_image_idx]
    
    # Run subprocess inside blender
    # Initialize class
    tb = TYOLTarget(scene_file, target_image_dir)
    # Generate tgt data
    tb.generate_tgt()