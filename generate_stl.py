#!/usr/bin/python3

import numpy as np;
import cv2;
import triangle;
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
  t = triangle.Triangle();
  points = np.squeeze(hull); # points.shape = (pts_num, 2)
  markers = np.ones((points.shape[0],)); # markers.shape = (pts_num)
  t.set_points(points, markers = markers.tolist());
  segments = zip(
    [i for i in range(points.shape[0])],
    [(i + 1) % points.shape[0] for i in range(points.shape[0])]
  ); # segments.shape = (seg_num, 2)
  t.set_segments(segments);
  t.triangulate(area = 0.01);
  triangles = t.get_triangles();
  '''
  img_contours = np.zeros(img.shape);
  cv2.drawContours(img, [hull,], -1, (0,255,0), 3);
  cv2.imshow('', img_contours);
  cv2.waitKey();
  '''
  
if __name__ == "__main__":

  from preprocess import preprocess;
  img = cv2.imread('/mnt/preview.bmp');
  leftfeet,rightfeet = preprocess(img);
  generate_stl(leftfeet[0]);
