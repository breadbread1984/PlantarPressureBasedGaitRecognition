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
    self.devices = ctx.query_devices();
    self.configs = list();
    self.filters = list();
    for device in self.devices:
      config = rs.config();
      config.enable_device(device.get_info(rs.camera_info.serial_number));
      config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30);
      config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30);
      self.configs.append(config);
      align = rs.align(rs.stream.color);
      spatial = rs.spatial_filter();
      spatial.set_option(rs.option.filter_magnitude, 5);
      spatial.set_option(rs.option.filter_smooth_alpha, 1);
      spatial.set_option(rs.option.filter_smooth_delta, 50);
      spatial.set_option(ts.option.holes_fill, 3);
      temporal = rs.temporal_filter();
      hole_filling = rs.hole_filling_filter();
      depth_to_disparity = rs.disparity_transform(True);
      disparity_to_depth = rs.disparity_transform(False);
      decimate = rs.decimation_filter();
      self.filters.append({'align': align, 'spatial': spatial, 'temporal': temporal, 'hole': hole_filling,
                           'disparity': depth_to_disparity, 'depth': disparity_to_depth, 'decimate': decimate});

  def info(self):
    
    retval = [{#'advanced_mode': device.get_info(rs.camera_info.advanced_mode),
               'asic_serial_number': device.get_info(rs.camera_info.asic_serial_number),
               'camera_locked': device.get_info(rs.camera_info.camera_locked),
               'debug_op_code': device.get_info(rs.camera_info.debug_op_code),
               'firmware_update_id': device.get_info(rs.camera_info.firmware_update_id),
               'firmware_version': device.get_info(rs.camera_info.firmware_version),
               'name': device.get_info(rs.camera_info.name),
               'physical_port': device.get_info(rs.camera_info.physical_port),
               'product_id': device.get_info(rs.camera_info.product_id),
               'product_line': device.get_info(rs.camera_info.product_line),
               #'recommended_firmware_version': device.get_info(rs.camera_info.recommended_firmware_version),
               'serial_number': device.get_info(rs.camera_info.serial_number),
               #'usb_type_descriptor': device.get_info(rs.camera_info.usb_type_descriptor)
               } for device in self.devices];
    return retval;

  def capture(self, cam_id):
    
    pipeline = rs.pipeline();
    pipeline.start(self.configs[cam_id]);
    # auto-exposure adjustment
    for i in range(5):
      pipeline.wait_for_frames();
    try:
      frames = pipeline.wait_for_frames();
      alignment = self.filters[cam_id]['align'].process(frames);
      depth_frame = alignment.get_depth_frame();
      depth_frame = self.filters[cam_id]['disparity'].process(depth_frame);
      depth_frame = self.filters[cam_id]['spatial'].process(depth_frame);
      depth_frame = self.filters[cam_id]['temporal'].process(depth_frame);
      depth_frame = self.filters[cam_id]['depth'].process(depth_frame);
      depth_frame = self.filters[cam_id]['hole'].process(depth_frame);
      color_frame = alignment.get_color_frame();
      if not depth_frame or not color_frame:
        pipeline.stop();
        return False, None, None;
      depth_image = np.asanyarray(depth_frame.get_data());
      color_image = np.asanyarray(color_frame.get_data());
      pipeline.stop();
      return True, depth_image.tolist(), color_image.tolist();
    except:
      pipeline.stop();
      return False, None, None;
  
  def point_cloud(self, cam_id):
      
    pipeline = rs.pipeline();
    pipeline.start(self.configs[cam_id]);
    # auto-exposure adjustment
    for i in range(5):
      pipeline.wait_for_frames();
    try:
      frames = pipeline.wait_for_frames();
      alignment = self.filters[cam_id]['align'].process(frames);
      depth_frame = alignment.get_depth_frame();
      color_frame = alignment.get_color_frame();
      depth_frame = decimate.process(depth_frame);
      self.filters[cam_id]['hole'].process(depth_frame);
      self.filters[cam_id]['decimate'].process(depth_frame);
      #depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics();
      depth_image = np.asanyarray(depth_frame.get_data());
      color_image = np.asanyarray(color_frame.get_data());
      depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data());
      pc = rs.pointcloud();
      points = pc.calculate(depth_frame);
      pc.map_to(color_frame);
      v = points.get_vertices();
      t = points.get_texture_coordinates();
      verts = np.asanyarray(v).view(np.float32).reshape(-1, 3); # xyz
      texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2); # uv
      # TODO

@celery.task(name = 'info', base = PlantarPressureWorker)
def info():

  return info.info();

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
