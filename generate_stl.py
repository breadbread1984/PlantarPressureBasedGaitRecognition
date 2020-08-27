#!/usr/bin/python3

import numpy as np;
import cv2;
import triangle;
import stl;
import tensorflow as tf;

def generate_stl(img, variable_thickness = 5, const_thickness = 1):

  if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
  else:
    gray = img.copy();
  # 1) mask
  ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY);
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
  all_contours = np.concatenate(contours, axis = 0);
  # 2) convex hull
  hull = cv2.convexHull(all_contours, False); # hull.shape = (num, 1, 2) in sequence of (x,y)
  # 3) generate triangle mesh for the polygon
  t = triangle.Triangle();
  points = np.squeeze(hull).astype('float32'); # points.shape = (pts_num, 2)
  markers = np.ones((points.shape[0],), dtype = np.int32); # markers.shape = (pts_num)
  t.set_points(points.tolist(), markers = markers.tolist());
  segments = list(zip(
    [i for i in range(points.shape[0])],
    [(i + 1) % points.shape[0] for i in range(points.shape[0])]
  )); # segments.shape = (seg_num, 2)
  t.set_segments(segments);
  t.set_holes([]);
  t.triangulate(area = 5);
  nodes = t.get_nodes();
  triangles = t.get_triangles();
  # 4) interpolate to get the pressure value at nodes
  # interpolate value at the float coordinates
  centers = np.array([node[0] for node in nodes], dtype = np.float32); # centers.shape = (node_num, 2) in sequence of (x,y)
  upperleft = np.floor(centers).astype('int32'); # upperleft = (min(x), min(y))
  downright = np.ceil(centers).astype('int32'); # downright = (max(x), max(y))
  upperright = np.concatenate([downright[:,0:1], upperleft[:,1:2]], axis = -1).astype('int32'); # upperright = (max(x), min(y))
  downleft = np.concatenate([upperleft[:,0:1], downright[:,1:2]], axis = -1).astype('int32'); # downleft = (min(x), max(y))
  # NOTE: gather with coordinates in sequence of (y, x), therefore we use reverse
  upperleft_value = tf.cast(tf.gather_nd(gray, tf.reverse(upperleft, axis = [-1])), dtype = tf.float64);
  downright_value = tf.cast(tf.gather_nd(gray, tf.reverse(downright, axis = [-1])), dtype = tf.float64);
  upperright_value = tf.cast(tf.gather_nd(gray, tf.reverse(upperright, axis = [-1])), dtype = tf.float64);
  downleft_value = tf.cast(tf.gather_nd(gray, tf.reverse(downleft, axis = [-1])), dtype = tf.float64);
  upperleft_weight = tf.math.exp(-tf.norm(upperleft - centers, axis = -1)); # upperleft_weight.shape = (node_num)
  downright_weight = tf.math.exp(-tf.norm(downright - centers, axis = -1)); # downright_weight.shape = (node_num)
  upperright_weight = tf.math.exp(-tf.norm(upperright - centers, axis = -1)); # upperright_weight.shape = (node_num)
  downleft_weight = tf.math.exp(-tf.norm(downleft - centers, axis = -1)); # downleft_weight.shape = (node_num)
  total = upperleft_weight + downright_weight + upperright_weight + downleft_weight;
  upperleft_weight = upperleft_weight / total;
  downright_weight = downright_weight / total;
  upperright_weight = upperright_weight / total;
  downleft_weight = downleft_weight / total;
  center_value = upperleft_value * upperleft_weight + downright_value * downright_weight + upperright_value * upperright_weight + downleft_value * downleft_weight; # center_value.shape = (node_num)
  # 5) output stl file
  # generate top mesh
  max_value = tf.math.reduce_max(center_value);
  min_value = tf.math.reduce_min(center_value);
  top_z = variable_thickness - (center_value - min_value) * variable_thickness / (max_value - min_value) + const_thickness; # top.shape = (node_num)
  top = tf.concat([centers, tf.expand_dims(top_z, axis = -1)], axis = -1); # top.shape = (node_num, 3)
  # generate bottom mesh
  bottom_z = tf.zeros_like(top); # bottom.shape = (node_num)
  bottom = tf.concat([centers, tf.expand_dims(bottom_z, axis = -1)], axis = -1); # bottom.shape = (node_num, 3)
  # generate edge
  
  '''
  img_contours = np.zeros(img.shape);
  cv2.drawContours(img, [hull,], -1, (0,255,0), 3);
  cv2.imshow('', img_contours);
  cv2.waitKey();
  '''
  
if __name__ == "__main__":

  from preprocess import preprocess;
  img = cv2.imread('preview.bmp');
  leftfeet,rightfeet = preprocess(img);
  generate_stl(leftfeet[0]);
