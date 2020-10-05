#!/usr/bin/python3

from os import mkdir, system;
from os.path import join, exists;
from shutil import rmtree;
from celery import Celery;
import numpy as np;
from scipy.misc import toimage;
from settings import *;

class Reconstruct(object):
  
  def __init__(self):

    self.worker = Celery('worker', backend = MESSAGE_QUEUE_URI, broker = MESSAGE_QUEUE_URI);

  def reconstruct(self, focal = 1536 openmvg_prefix='/root/opt/openmvg', openmvs_prefix='/root/opt/openmvs'):

    captured = self.__capture();
    sequence = 0;
    if exists('captured'): rmtree('captured');
    else: mkdir('captured');
    for depth, color in captured:
      toimage(depth, cmin = CLIPPED_LOW, cmax = CLIPPED_HIGH).save(join('captured', str(sequence).zfill(3) + '_mask.png'));
      toimage(depth).save(join('captured', str(sequence).zfill(3) + '.png'));
    # generate image list
    try:
      system(join(openmvg_prefix, 'bin', 'openMVG_main_SfMInit_ImageListing') + \
             ' -d ' + join(openmvg_prefix, 'share', 'openMVG', 'sensor_width_camera_database.txt') + \
             ' -i ' + 'captured/' + \
             ' -o ' + 'matches/' + \
             str(focal));
    except:
      print('openMVG_main_SfMInit_ImageListing failed!');
      return;
    # TODO: call openMVG openMVS
    
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
