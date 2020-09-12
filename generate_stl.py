#!/usr/bin/python3

import numpy as np;
import cv2;
import triangle;
from stl import mesh;
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
  t.triangulate(area = 0.1);
  t.refine(area_ratio=1.5);
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
  bottom_z = tf.zeros_like(top_z); # bottom.shape = (node_num)
  bottom = tf.concat([centers, tf.expand_dims(bottom_z, axis = -1)], axis = -1); # bottom.shape = (node_num, 3)
  # generate edge
  top_edge = tf.concat([tf.cast(tf.squeeze(hull), dtype = tf.float64), (variable_thickness + const_thickness) * tf.ones((hull.shape[0], 1), dtype = tf.float64)], axis = -1); # top_edge.shape = (pts_num, 3)
  bottom_edge = tf.concat([tf.cast(tf.squeeze(hull), dtype = tf.float64), tf.zeros((hull.shape[0], 1), dtype = tf.float64)], axis = -1); # bottom_edge.shape = (pts_num, 3)
  edge = tf.concat([top_edge, bottom_edge], axis = 0); # edge.shape = (pts_num * 2, 3)
  edge_faces = list();
  for i in range(hull.shape[0]):
    edge_faces.append((i, i + 1, (i + hull.shape[0]) % (2*hull.shape[0])));
    edge_faces.append((i + 1, (i + hull.shape[0]) % (2*hull.shape[0]), (i + hull.shape[0] + 1) % (2*hull.shape[0])));
  vertices = tf.concat([top, bottom, edge], axis = 0); # vertices.shape = (node_num * 2, 3)
  top_faces = tf.constant([face[0] for face in triangles], dtype = tf.int32); # top_faces.shape = (face_num, 3)
  bottom_faces = top_faces + top.shape[0]; # bottom_faces.shape = (face_num, 3)
  edge_faces = tf.constant(edge_faces, dtype = tf.int32) + top.shape[0] + bottom.shape[0];
  faces = tf.concat([top_faces, bottom_faces, edge_faces], axis = 0); # faces.shape = (face_num * 2 + edge_face_num, 3)
  solid = mesh.Mesh(np.zeros(faces.shape[0], dtype = mesh.Mesh.dtype));
  for i, face in enumerate(faces):
    for j in range(3):
      solid.vectors[i][j] = vertices[face[j], :];
  solid.save('solid.stl');
  
if __name__ == "__main__":

  from preprocess import preprocess;
  img = cv2.imread('preview.bmp');
  leftfeet,rightfeet = preprocess(img);
  generate_stl(leftfeet[0]);
