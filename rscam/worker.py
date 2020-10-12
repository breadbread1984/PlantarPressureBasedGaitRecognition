#!/usr/bin/python3

from celery import Celery, Task;
from celery.signals import after_setup_logger;
import logging;
import pyrealsense2 as rs;
import numpy as np;
import cv2;
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

  def size(self, cam_id):
      
    pipeline = rs.pipeline();
    pipeline.start(self.configs[cam_id]);
    profile = pipeline.get_active_profile();
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth));
    depth_intrinsics = depth_profile.get_intrinsics();
    w, h = depth_intrinsics.width, depth_intrinsics.height;
    pipeline.stop();
    return w, h;

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
  
  def get_point_cloud(self, cam_id):
      
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
      verts = np.asanyarray(v).view(np.float32).reshape(-1, 3); # xyz.shape = (n, 3)
      texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2); # uv.shape = (n, 2) in relative float number
      return verts, texcoords;

  def pointcloud(self, out, verts, texcoords, color, painter = True, translation = np.array([0, 0, -1], dtype = np.float32), pitch = 0, yaw = 0, distance = 2):

    # out: image where the voxels are painted
    # verts: the coordinates of voxel in camera coordinate system
    # texcoordinate: the coordinates of voxel in image coordinate system
    # color: the captured image
    if painter:
      v = self.view(verts, translation, pitch, yaw, distance); # get coordinate of verts in world coordinate system
      s = v[:, 2].argsort()[::-1]; # sort coordinates with respect to z value in descent order in order to draw further voxel first
      proj = project(v[s]); # get homogeneous coordinate
    else:
      # not sorting to the voxels, therefore the voxels are not draw in order
      proj = project(self.view(verts, translation, pitch, yaw, distance));
    h, w = out.shape[:2];
    j, i = proj.astype(np.uint32); # get u,v of homogeneous coordinate
    # mask of visible voxel in image area
    im = (i >= 0) & (i < h);
    jm = (j >= 0) & (j < w);
    m = im & jm;
    cw, ch = color.shape[:2][::-1]; # get rgb captured image size
    # turn the texcoordinate into absolute coordinate
    if painter:
      v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T;
    else:
      v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T;
    # clip texcoordinate within captured image area
    np.clip(u, 0, ch-1, out=u);
    np.clip(v, 0, cw-1, out=v);
    # paint the output image
    out[i[m], j[m]] = color[u[m], v[m]];
    return out;

  def project(self, out, v):

    # project coordinate in camera coordinate system to homogeneous coordinate in image coordinate system
    # NOTE: out.shape = (h, w, 3) v.shape = (n, 3)
    h, w = out.shape[:2];
    view_aspect = float(h) / w;
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
      # (x,y)/z * (h,h) + (w/2,h/2)
      proj = v[:, :-1] / v[:, -1:] * (w * view_aspect, h) + (w/2.0, h/2.0);
    znear = 0.03;
    # ignore the points with small z value
    proj[v[:, 2] < znear] = np.nan;
    return proj;

  def view(self, v, translation = np.array([0, 0, -1], dtype = np.float32), pitch = 0, yaw = 0, distance = 2):

    # NOTE: coordinate in old camera coordinate system -> coordinate in new camera coordinate system
    # v: object coordinate in the old camera (RealSense camera) coordinate system
    # translation: the translation of the new camera (virtual camera) with respect to the old camera (RealSense camera)
    # (0, 0, distance): the translation of the pivot with respect to the new camera (virtual camera)
    # pitch: pitch angle with respect to the old camera (RealSense camera) coordinate system
    # yaw: yaw angle with respect to the old camera (RealSense camera) coordinate system
    # output: object coordinate in the new camera (virtual camera) coordinate system
    # NOTE: because, the translation of the pivot with respect to the old camera is fixed
    # when change distance, the translation of the new camera have to be changed accordingly
    pivot = translation + np.array((0, 0, distance), dtype = np.float32); # translation of the pivot with respect to the old camera (RealSense camera)
    Rx, _ = cv2.Rodrigues((pitch, 0, 0)); # euler angles -> rotation matrix
    Ry, _ = cv2.Rodrigues((0, yaw, 0)); # euler angles -> rotation matrix
    rotation = np.dot(Ry, Rx).astype(np.float32); # merged rotation matrix
    return np.dot(v - pivot, rotation) + pivot - translation; # coordinate in new camera (virtual camera) coordinate system

@celery.task(name = 'info', base = PlantarPressureWorker)
def info():

  return info.info();

@celery.task(name = 'size', base = PlantarPressureWorker)
def size():

  return size.size();

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
