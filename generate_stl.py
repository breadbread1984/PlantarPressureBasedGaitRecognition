#!/usr/bin/python3

import numpy as np;
import cv2;
import pymesh;
#import pygmsh;
import stl;

def generate_stl(img):

  if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
  else:
    gray = img.copy();
  # 1) mask
  ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY);
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
  all_contours = np.concatenate(contours, axis = 0);
  # 2) convex hull
  hull = cv2.convexHull(all_contours, False); # hull.shape = (num, 1, 2)
  # 3) generate triangle mesh for the polygon
  vertices = np.squeeze(hull);
  tri = pymesh.triangle();
  tri.max_area = 0.05;
  tri.split_boundary = True;
  tri.verbosity = 0;
  tri.run();
  mesh = tri.mesh;
  '''
  geom = pygmsh.built_in.Geometry();
  coords = np.squeeze(hull); # coords.shape = (num, 2)
  coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis = -1); # coords.shape = (num, 3)
  poly = geom.add_polygon(coords.tolist(), lcar = 0.05);
  geom.extrude(poly, translation_axis = (0,0,1), rotation_axis = (0,0,1), point_on_axis = (0,0,0), angle = 0);
  mesh = pygmsh.generate_mesh(geom);
  print(mesh)
  '''
  '''
  img_contours = np.zeros(img.shape);
  cv2.drawContours(img, [hull,], -1, (0,255,0), 3);
  cv2.imshow('', img_contours);
  cv2.waitKey();
  '''
  
if __name__ == "__main__":

  from preprocess import preprocess;
  img = cv2.imread('/mnt/c/Users/Lenovo/Downloads/preview.bmp');
  leftfeet,rightfeet = preprocess(img);
  generate_stl(leftfeet[0]);
