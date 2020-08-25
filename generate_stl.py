#!/usr/bin/python3

import cv2;
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
  hull = cv2.convexHull(all_contours, False);
  
  img_contours = np.zeros(img.shape);
  cv2.drawContours(img, [hull,], -1, (0,255,0), 3);
  cv2.imshow('', img_contours);
  cv2.waitKey();
