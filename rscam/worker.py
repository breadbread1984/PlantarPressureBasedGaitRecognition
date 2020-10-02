#!/usr/bin/python3

from celery import Celery, Task;
from celery.signals import after_setup_logger;
import logging;
import pyrealsense2 as rs;
import numpy as np;
from settings import *;

celery = Celery('worker', backend = MESSAGE_QUEUE_URI, broker = MESSAGE_QUEUE_URI);
celery.conf.broker_transport_options = {'visibility_timeout': 3600 * 10};

logger = logging.getLogger(__name__);

@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
  
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s');
  fh = logging.FileHandler('logs.log');
  fh.setFormatter(formater);
  logger.addHandler(fh);

class PlantarPressureWorker(Task):

  def __init__(self):

    ctx = rs.context();
    devices = ctx.query_devices();
    self.configs = list();
    for device in devices:
      config = rs.config();
      config.enable_device(device.get_info(rs.camera_info.serial_number));
      config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30);
      config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30);
      self.configs.append(config);

  def count(self):
    
    return len(self.configs);

  def capture(self, cam_id):
    
    pipeline = rs.pipeline();
    pipeline.start(self.configs[cam_id]);
    try:
      frames = pipeline.wait_for_frames();
      depth_frame = frames.get_depth_frame();
      color_frame = frames.get_color_frame();
      if not depth_frame or not color_frame:
        pipeline.stop();
        return False, None, None;
      depth_image = np.asanyarray(depth_frame.get_data());
      color_image = np.asanyarray(color_frame.get_data());
      pipeline.stop();
      return True, depth_image, color_image;
    except:
      pipeline.stop();
      return False, None, None;

@celery.task(name = 'count', base = PlantarPressureWorker)
def count():

  return count.count();

@celery.task(name = 'capture', base = PlantarPressureWorker)
def capture(cam_id):

  return capture.capture(cam_id);

if __name__ == "__main__":

  from os import system;
  import signal;
  from time import sleep;
  system('bash start_worker.sh');
  def stop_worker(sig, frame):
    system('bash stop_worker.sh');
    exit(0);
  signal.signal(signal.SIGINT, stop_worker);
  while True:
    sleep(60);
