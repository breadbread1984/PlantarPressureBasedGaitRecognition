#!/usr/bin/python3

import pickle;
import numpy as np;
import cv2;
from generate_stl import generate_stl;

def generate(gray):

  gray = cv2.resize(gray, (250,250), interpolation = cv2.INTER_CUBIC);
  leftfeet, rightfeet = gray[:, :125], gray[:, 125:];
  generate_stl(leftfeet);

if __name__ == "__main__":

  with open('dataset.pkl', 'rb') as f:
    dataset = pickle.loads(f.read());
  gray = dataset[0];
  generate(gray);

