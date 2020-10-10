#!/usr/bin/python3

from math import sqrt, copysign;
from celery import Celery;
import numpy as np;
import cv2;
import tensorflow as tf;
import pyrealsense2 as rs;
from settings import *;

class VisualizeAdjustment(object):

  def __init__(self):

    self.worker = Celery('worker', backend = MESSAGE_QUEUE_URI, broker = MESSAGE_QUEUE_URI);
    self.distance = 2;
    # translations: the translations of the new camera with respect to the old cameras
    self.translations = list();
    infos = self.worker.send_task(name = 'info', args = []);
    assert len(infos) == 4;
    # set the new camera at right back of the first realsense camera
    self.translations.append(np.array([0, 0, -1], dtype = np.float32)); # south
    self.translations.append(np.array([-2, 0, 1], dtype = np.float32)); # east
    self.translations.append(np.array([0, 0, 3], dtype = np.float32)); # north
    self.translations.append(np.array([2, 0, 1], dtype = np.float32)); # west

  def view(self, v, cam_id, pitch = 0, yaw = 0):

    # v.shape = (n, 3)
    # cam_id.shape = (n)
    pivots = list();
    for translation in self.translations:
      pivots.append(translation + np.array((0, 0, self.distance), dtype = np.float32));
    pivots = np.array(pivots); # pivots.shape = (4, 3)
    Rx, _ = cv2.Rodrigues((pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, yaw, 0));
    rotation = np.dot(Ry, Rx).astype(np.float32);
    pivots = tf.gather(pivots, cam_id).numpy(); # pivots.shape = (n, 3)
    translations = tf.gather(self.translations, cam_id).numpy(); # translations.shape = (n, 3)
    return np.dot(v - pivots, rotation) + pivots - translations; # shape = (n, 3)

  def visualize(self):

    depth_profile = rs.video_stream_profile
    out = np.empty((), dtype = np.uint8);

  def set_distance(self, distance):

    self.distance = distance;

if __name__ == "__main__":

  va = VisualizeAdjustment();
  prev_mouse = (0, 0);
  translations = list();
  def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:

      dx, dy = x - prev_mouse[0], y - prev_mouse[1];
      dz = sqrt(dx**2 + dy**2) + copysign(0.01, -dy);
      

    prev_mouse = (x, y);
      
  while True:
    image = va.visualize();
    cv2.imshow('show', image);
    key = cv2.waitKey(1);

