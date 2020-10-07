#!/usr/bin/python3

from enum import Enum;
from os import mkdir, system;
from os.path import join, exists;
from shutil import rmtree;
from celery import Celery;
import numpy as np;
from scipy.misc import toimage;
from settings import *;

class DescriberPreset(Enum):
  NORMAL = 'NORMAL';
  HIGH = 'HIGH';
  ULTRA = 'ULTRA';
  
class DescriberMethod(Enum):
  SIFT = 'SIFT';
  AKAZE_FLOAT = 'AKAZE_FLOAT';
  AKAZE_MLDB = 'AKAZE_MLDB';

class GeometricModel(Enum):
  Fundamental: 'f';
  Essential: 'e';
  Homography: 'h';
  
class NearestMatchingMethod(Enum):
  AUTO: 'AUTO';
  BRUTEFORCEL2: 'BRUTEFORCEL2';
  ANNL2: 'ANNL2';
  CASCADEHASHINGL2: 'CASCADEHASHINGL2';
  FASTCASCADEHASHINGL2: 'FASTCASCADEHASHINGL2';
  BRUTEFORCEHAMMING: 'BRUTEFORCEHAMMING';

class Reconstruct(object):
  
  def __init__(self):

    self.worker = Celery('worker', backend = MESSAGE_QUEUE_URI, broker = MESSAGE_QUEUE_URI);

  def reconstruct(self, focal = 1536, 
                  describerPreset: DescriberPreset = DescriberPreset.NORMAL,
                  describerMethod: DescriberMethod = DescriberMethod.SIFT,
                  ratio = 0.8,
                  geometricModel: GeometricModel = GeometricModel.Fundamental,
                  nearestMatchingMethod: NearestMatchingMethod.FASTCASCADEHASHINGL2,
                  openmvg_prefix='/root/opt/openmvg', openmvs_prefix='/root/opt/openmvs'):

    captured = self.__capture();
    # 0) save color, mask, masked color images
    sequence = 0;
    if exists('captured'): rmtree('captured');
    mkdir('captured');
    if exists('images'): rmtree('images');
    mkdir('images');
    for depth, color in captured:
      toimage(depth, cmin = CLIPPED_LOW, cmax = CLIPPED_HIGH).save(join('captured', str(sequence).zfill(3) + '_mask.png'));
      toimage(color).save(join('captured', str(sequence).zfill(3) + '.png'));
      masked_color = color.copy();
      masked_color[np.bitwise_or(depth == CLIPPED_HIGH, depth == CLIPPED_LOW)] = np.zeros((color.shape[-1]));
      toimage(masked_color).save(join('images', str(sequence).zfill(3) + '.png'));
    # 1) generate image list
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_SfMInit_ImageListing') + \
             ' -d ' + join(openmvg_prefix, 'share', 'openMVG', 'sensor_width_camera_database.txt') + \
             ' -i ' + 'images/' + \
             ' -o ' + 'matches/' + \
             ' ' + str(focal));
    except:
      print('openMVG_main_SfMInit_ImageListing failed!');
      return;
    # 2) compute features
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_ComputeFeatures') + \
             ' -i ' + 'matches/sfm_data.json' + \
             ' -o ' + 'matches' + \
             ' ' + describerPreset.value + \
             ' ' + describerMethod.value);
    except:
      print('openMVG_main_ComputeFeatures failed!');
      return;
    # 3) match features
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_ComputeMatches') + \
            ' -i ' + 'matches/sfm_data.json' + \
            ' -o ' + 'matches' + \
            ' ' + ratio + \
            ' ' + geometricModel.value + \
            ' ' + nearestMatchingMethod.value);
    except:
      print('openMVG_main_ComputeFeatures failed!');
      return;
    # 4) structure from motion
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_IncrementalSfM') + \
             ' -i ' + 'matches/sfm_data.json' + \
             ' -m ' + 'matches' + \
             ' -o ' + 'out_Incremental_Reconstruction');
    except:
      print('openMVG_main_IncrementalSfM failed!');
      return;
    # 5) convert openMVG to openMVS
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_openMVG2openMVS') + \
             ' -i ' + 'out_Incremental_Reconstruction/sfm_data.bin' + \
             ' -o ' + 'scene.mvs');
    except:
      print('openMVG_main_openMVG2openMVS failed!');
      return;
    # 6) densify point cloud
    try:
      system(join(openmvs_prefix, 'bin', 'DensifyPointCloud') + \
             ' ' + 'scene.mvs');
    except:
      print('DensifyPointCloud failed!');
      return;
    # 7) reconstruct mesh
    try:
      system(join(openmvs_prefix, 'bin', 'ReconstructMesh') + \
             ' ' + 'scene_dense.mvs');
    except:
      print('ReconstructMesh failed!');
      return;
    # 8) refine mesh
    try:
      system(join(openmvs_prefix, 'bin', 'RefineMesh') + \
             ' ' + 'scene_dense_mesh.mvs' + \
             ' ' + '--max-face-area 16');
    except:
      print('RefineMesh failed!');
      return;
    # 8) texture mesh
    try:
      system(join(openmvs_prefix, 'bin', 'TextureMesh') + \
             ' ' + 'scene_dense_mesh_refine.mvs');
    except:
      print('TextureMesh failed!');
      return;
    
  def __masked(self, depth, distance = CLIPPING_DISTANCE):

    clipping_distance = distance / depth_scale;
    depth[depth < clipping_distance] = CLIPPED_LOW;
    depth[depth > clipping_distance] = CLIPPED_HIGH;
    return depth;

  def __capture(self):

    result = self.worker.send_task(name = 'info', args = []);
    devices = result.get();
    assert len(devices) == 4;
    retval = list();
    for i in range(len(devices)):
      result = self.worker.send_task(name = 'capture', args = [i]);
      succeed, depth, color = result.get();
      if succeed == False:
        return False;
      depth = np.array(depth);
      color = np.array(color);
      retval.append((self.__masked(depth), color));
    return retval;

if __name__ == "__main__":

  Reconstruct recon;
  recon.reconstruct();
