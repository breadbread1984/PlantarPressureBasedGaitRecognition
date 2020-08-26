#!/usr/bin/python3

import sys;
from os.path import join, exists, isdir;
import pickle;
import numpy as np;
import cv2;
import tensorflow as tf;
from preprocess import preprocess;

def main(inputdir):

  leftwriter = tf.io.TFRecordWriter("leftfeet.tfrecord");
  rightwriter = tf.io.TFRecordWriter("rightfeet.tfrecord");
  count = 0;
  ids = dict();
  for f in listdir(inputdir):
    if isdir(join(inputdir,f)):
      imgs = list();
      label = count;
      # map label to identity
      ids[label] = f;
      count += 1;
      # only visit directory under given directory
      for img in listdir(join(directory,f)):
        if False == isdir(join(directory, f, img)):
          # visit every image under directory
          image = cv2.imread(join(directory, f, img));
          if image is None:
            print('can\'t open file '+ join(directory, f, img));
            continue;
          # crop feet from image
          leftfeet, rightfeet = preprocess(image, (224, 224));
          # store feet into dataset file
          for leftfoot in leftfeet:
            trainsample = tf.train.Example(features = tf.train.Features(
              feature = {
                'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = leftfoot.tobytes())),
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
              }
            ));
            leftwriter.write(trainsample.SerializeToString());
          for rightfoot in rightfeet:
            trainsample = tf.train.Example(features = tf.train.Features(
              feature = {
                'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [rightfoot.tobytes()])),
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
              }
            ));
  leftwriter.close();
  rightwriter.close();
  with open('ids.pkl', 'wb') as f:
    f.write(pickle.dumps(ids));
  return True;

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + " <input dir>");
    exit(1);
  main(sys.argv[1]);
