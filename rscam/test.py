#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from shutil import rmtree;
import cv2;
import pyrealsense2 as rs;
import numpy as np;
from settings import *;

def main():

  ctx = rs.context();
  devices = ctx.query_devices();
  device = devices[0];
  # print device info
  #print('advanced_mode: %s' % device.get_info(rs.camera_info.advanced_mode));
  print('asic_serial_number: %s' % device.get_info(rs.camera_info.asic_serial_number));
  print('camera_locked: %s' % device.get_info(rs.camera_info.camera_locked));
  print('debug_op_code: %s' % device.get_info(rs.camera_info.debug_op_code));
  print('firmware_update_id: %s' % device.get_info(rs.camera_info.firmware_update_id));
  print('firmware_version: %s' % device.get_info(rs.camera_info.firmware_version));
  print('name: %s' % device.get_info(rs.camera_info.name));
  print('physical_port: %s' % device.get_info(rs.camera_info.physical_port));
  print('product_id: %s' % device.get_info(rs.camera_info.product_id));
  print('product_line: %s' % device.get_info(rs.camera_info.product_line));
  #print('recommended_firmware_version: %s' % device.get_info(rs.camera_info.recommended_firmware_version));
  print('serial_number: %s' % device.get_info(rs.camera_info.serial_number));
  #print('usb_type_descriptor: %s' % device.get_info(rs.camera_info.usb_type_descriptor));
  
  filters = dict();
  filters['align'] = rs.align(rs.stream.color);
  filters['spatial'] = rs.spatial_filter();
  filters['spatial'].set_option(rs.option.filter_magnitude, 5);
  filters['spatial'].set_option(rs.option.filter_smooth_alpha, 1);
  filters['spatial'].set_option(rs.option.filter_smooth_delta, 50);
  filters['spatial'].set_option(rs.option.holes_fill, 3);
  filters['temporal'] = rs.temporal_filter();
  filters['hole'] = rs.hole_filling_filter();
  filters['disparity'] = rs.disparity_transform(True);
  filters['depth'] = rs.disparity_transform(False);
  config = rs.config();
  config.enable_device(device.get_info(rs.camera_info.serial_number));
  config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30);
  config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30);
  pipeline = rs.pipeline();
  if exists('captured'): rmtree('captured');
  if exists('images'): rmtree('images');
  mkdir('captured');
  mkdir('images');
  sequence = 0;
  while True:
    ch = input('press to capture');
    pipeline.start(config);
    for i in range(5):
      pipeline.wait_for_frames();
    frames = pipeline.wait_for_frames();
    alignment = filters['align'].process(frames);
    depth_frame = alignment.get_depth_frame();
    #depth_frame = filters['disparity'].process(depth_frame);
    #depth_frame = filters['spatial'].process(depth_frame);
    #depth_frame = filters['temporal'].process(depth_frame);
    #depth_frame = filters['depth'].process(depth_frame);
    depth_frame = filters['hole'].process(depth_frame);
    color_frame = alignment.get_color_frame();
    if not depth_frame or not color_frame:
      print("failed to capture!");
      continue;
    depth_image = np.asanyarray(depth_frame.get_data());
    print(np.min(depth_image), np.max(depth_image));
    mask = masked(depth_image.copy());
    color_image = np.asanyarray(color_frame.get_data());
    cv2.imwrite(join('captured', str(sequence).zfill(3) + '_mask.png'), cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03), cv2.COLORMAP_JET));
    cv2.imwrite(join('captured', str(sequence).zfill(3) + '.png'), color_image);
    masked_color = color_image.copy();
    masked_color[mask == CLIPPED_HIGH] = np.zeros((color_image.shape[-1]));
    cv2.imwrite(join('images', str(sequence).zfill(3) + '.png'), masked_color);
    pipeline.stop();
    sequence += 1;

def masked(depth, distance = CLIPPING_DISTANCE):

  clipping_distance = distance / depth_scale;
  depth[depth < clipping_distance] = CLIPPED_LOW;
  depth[depth > clipping_distance] = CLIPPED_HIGH;
  return depth;

if __name__ == "__main__":

  main();
