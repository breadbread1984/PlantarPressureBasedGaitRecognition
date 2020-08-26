#!/usr/bin/python3

import sys;
import numpy as np;
import cv2;
import tensorflow as tf;

def sift(img_path):

  img = cv2.imread(img_path);
  if img is None: return False;
  img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])));
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
  sift = cv2.xfeatures2d.SIFT_create();
  kps = sift.detect(gray, None);
  img = cv2.drawKeypoints(img, kps, outImage = None, color = (0,255,0));
  pts = np.array([kps[idx].pt for idx in range(0, len(kps))], dtype = np.float32).reshape(-1, 2);
  return pts, img.shape;

def pressure(pts, shape):

  xy = tf.stack(
    [
      tf.tile(tf.reshape(tf.range(tf.cast(shape[1], dtype = tf.float32), dtype = tf.float32), (1, shape[1])), (shape[0], 1)),
      tf.tile(tf.reshape(tf.range(tf.cast(shape[0], dtype = tf.float32), dtype = tf.float32), (shape[0], 1)), (1, shape[1]))
    ], axis = -1); # xy.shape = (h, w, 2) in sequence of (x,y)
  xy = tf.expand_dims(xy, axis = -2); # xy.shape = (h, w, 1, 2)
  pts = tf.reshape(pts, (1,1,-1,2)); # pts.shape = (1, 1, n, 2)
  dist = tf.norm(xy - pts, ord = 'euclidean', axis = -1); # dist.shape = (h,w,n)
  heat = 0.2 * tf.math.exp(-dist ** 2 / 100); # heat.shape = (h,w,n)
  heat_map = tf.math.reduce_sum(heat, axis = -1); # heat_map.shape = (h, w)
  return heat_map.numpy();

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <image>");
    exit();
  pts, shape = sift(sys.argv[1]);
  heatmap = pressure(pts, shape);
  import matplotlib.pyplot as plt;
  fig = plt.figure();
  plt.imshow(heatmap, cmap = 'hot');
  fig.savefig('plot.pdf');
  
