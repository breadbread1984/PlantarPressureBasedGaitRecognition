#!/usr/bin/python3

import os;
import tensorflow as tf;
from MobileNetV3 import create_mobilenet_v3;

input_shape = (224,224,3);
batch_size = 100;
num_ids = 100; # number of distinct identities

def parse_function(serialized_example):
    
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string, default_value = ''),
      'label': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0)
    }
  );
  data = tf.io.decode_raw(feature['data'], out_type = tf.uint8);
  data = tf.reshape(data, input_shape);
  data = tf.cast(data, dtype = tf.float32);
  label = tf.cast(feature['label'], dtype = tf.int32);
  return data, label;

def main():

  # create mobilenet v3
  model = create_mobilenet_v3(tf.keras.Input(input_shape), num_classes = num_ids);
  optimizer = tf.keras.optimizers.Adam(1e-3);
  # load dataset
  trainset = tf.data.TFRecordDataset(os.path.join('dataset','trainset.tfrecord')).map(parse_function).shuffle(batch_size).batch(batch_size);
  # restore checkpoint
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer, optimizer_step = optimizer.iterations);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
  # create log
  log = tf.summary.create_file_writer('checkpoint');
  # train model
  print("training ...");
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    for (images, labels) in trainset:
      with tf.GradientTape() as tape:
        logits = model(images);
        loss = tf.keras.losses.SparseCategoricalCrossentropy(labels, logits);
        avg_loss.update_state(loss);
      # write log
      if tf.equal(optimizer.iterations % 100, 0):
        with log.as_default():
          tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
        print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.results()));
        avg_loss.reset_states();
      # apply gradients
      grads = tape.gradient(loss, model.trainable_variables);
      optimizer.apply_gradients(zip(grads, model.trainable_variables));
    #save model once every epoch
    checkpoint.save(os.path.join('checkpoint','ckpt'));
    if loss < 1e-3: break;
  # save the network structure with weights
  model.save('mobilenetv3.h5');

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  main();
