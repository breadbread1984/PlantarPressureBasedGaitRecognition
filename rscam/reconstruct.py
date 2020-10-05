#!/usr/bin/python3

from celery import Celery;
import numpy as np;
from settings import *;

class Reconstruct(object):
  
  def __init__(self):

    self.worker = Celery('worker', backend = MESSAGE_QUEUE_URI, broker = MESSAGE_QUEUE_URI);

  def reconstruct(self):

    

    
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
      retval.append((depth, color))'
    return retval;
