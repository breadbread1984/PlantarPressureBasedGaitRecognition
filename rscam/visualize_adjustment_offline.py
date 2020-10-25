#!/usr/bin/python3

from math import sqrt, copysign, pi;
import numpy as np;
import cv2;
import pyrealsense2 as rs;
import tensorflow as tf;
from settings import *;

class VisualizeAdjustment(object):

  def __init__(self):

    # translations: the translations of the new camera with respect to the old cameras
    self.translations = list();
    # load capture from four directions
    self.cams = {'south': {'verts': np.load('capture/south/verts.npy'), 
                           'texcoords': np.load('capture/south/texcoords.npy'), 
                           'color': np.load('capture/south/color_image.npy'), 
                           'depth': np.load('capture/south/depth_image.npy')},
                 'east': {'verts': np.load('capture/east/verts.npy'), 
                          'texcoords': np.load('capture/east/texcoords.npy'), 
                          'color': np.load('capture/east/color_image.npy'), 
                          'depth': np.load('capture/east/depth_image.npy')},
                 'north': {'verts': np.load('capture/north/verts.npy'), 
                           'texcoords': np.load('capture/north/texcoords.npy'), 
                           'color': np.load('capture/north/color_image.npy'), 
                           'depth': np.load('capture/north/depth_image.npy')},
                 'west': {'verts': np.load('capture/west/verts.npy'), 
                          'texcoords': np.load('capture/west/texcoords.npy'), 
                          'color': np.load('capture/west/color_image.npy'), 
                          'depth': np.load('capture/west/depth_image.npy')}};
    self.cam_names = ['south', 'east', 'north', 'west'];
    # the distance from the new camera to the pivot
    self.distance = 2;
    # the translation of the new camera with respect to the old camera coordinate system
    self.translations.append(np.array([0, 0, -1], dtype = np.float32)); # south
    self.translations.append(np.array([-2, 0, 1], dtype = np.float32)); # east
    self.translations.append(np.array([0, 0, 3], dtype = np.float32)); # north
    self.translations.append(np.array([2, 0, 1], dtype = np.float32)); # west
    self.pitch = 0;
    self.yaw = 0;

  def view(self, v, cam_id):

    # INFO: this function transform cloud points from physical camera coordinate systems to the virtual camera coordinate system
    # NOTE: translation: the translation of the new camera (virtual camera) with respect to the old camera (RealSense camera)
    # NOTE: (x, y, z): the translation of the pivot with respect to the new camera (virtual camera)
    # NOTE: the pivot is fixed with respect to the old cameras
    # v.shape = (n, 3) the coordinate in a physical camera coordinate system
    # cam_id.shape = (n) the index of the camera coordinate system which the cloud point belongs to.
    pivots = list();
    rotations = list();
    # south
    pivots.append(self.translations[0] + np.array((0, 0, self.distance), dtype = np.float32));
    Rx, _ = cv2.Rodrigues((self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw, 0));
    rotations.append(np.dot(Ry, Rx).astype(np.float32));
    # east
    pivots.append(self.translations[1] + np.array((self.distance, 0, 0), dtype = np.float32));
    Rx, _ = cv2.Rodrigues((self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2, 0));
    rotations.append(np.dot(Ry, Rx).astype(np.float32))
    # north
    pivots.append(self.translations[2] + np.array((0, 0, -self.distance), dtype = np.float32));
    Rx, _ = cv2.Rodrigues((self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2 * 2, 0));
    rotations.append(np.dot(Ry, Rx).astype(np.float32));
    # west
    pivots.append(self.translations[3] + np.array((-self.distance, 0, 0), dtype = np.float32));
    Rx, _ = cv2.Rodrigues((self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2 * 3, 0));
    rotations.append(np.dot(Ry, Rx).astype(np.float32));
    pivots = np.array(pivots); # pivots.shape = (4, 3)
    rotations = np.array(rotations); # rotations.shape = (4, 3, 3)
    pivots = tf.gather(pivots, cam_id); # pivots.shape = (n, 3)
    rotations = tf.gather(rotations, cam_id); # rotations.shape = (n, 3, 3)
    translations = tf.gather(self.translations, cam_id); # translations.shape = (n, 3)
    transformed = tf.squeeze(tf.linalg.matmul(tf.expand_dims(v - pivots, axis = 1), rotations), axis = 1) + pivots - translations;
    return transformed.numpy(); # shape = (n, 3)

  def project(self, size, v):

    # INFO: this function map coordinate in camera coordinate system to homogeneous coordinate in image coordinate system
    # size.shape = (2), the size of image in sequence of w, h
    # v.shape = (n, 3), the coordinate in the virtual camera coordinate system
    w, h = size;
    view_aspect = float(h) / w;
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
      proj = v[:, :-1] / v[:, -1:] + (w * view_aspect, h) + (w / 2.0, h / 2.0);
    znear = 0.03;
    # ignore the points with small z value
    proj[v[:, 2] < znear] = np.nan;
    return proj;

  def visualize(self):

    # INFO: this function visualize the point cloud into an image
    captures = list();
    pc = rs.pointcloud();
    # 1) get point cloud from four cameras
    total_verts = list();
    total_cam_id = list();
    total_texcoords = list();
    total_colors = list();
    for i in range(4):
      cam_name = self.cam_names[i];
      verts, texcoords, depth, color = self.cams[cam_name]['verts'], self.cams[cam_name]['texcoords'], self.cams[cam_name]['depth'], self.cams[cam_name]['color'];
      total_verts.append(verts); # verts.shape = (320 * 240,3)
      total_cam_id.append(i * np.ones((verts.shape[0]), dtype = np.int32)); # shape = (320 * 240)
      total_texcoords.append(texcoords); # texcoords.shape = (320 * 240, 2)
      total_colors.append(color); # color.shape = (480, 640, 3)
    total_verts = np.concatenate(total_verts, axis = 0); # total_verts.shape = (320 * 240, 3)
    total_cam_id = np.concatenate(total_cam_id, axis = 0); # total_cam_id.shape = (320 * 240)
    total_texcoords = np.concatenate(total_texcoords, axis = 0); # total_texcoords.shape = (320 * 240, 2)
    total_colors = np.stack(total_colors, axis = 0); # total_colors.shape = (4, 480, 640, 3)
    # 2) project point cloud from four physical cameras to the virtual camera
    v = self.view(total_verts, total_cam_id);
    s = v[:, 2].argsort()[::-1]; # sort coordinates according to z value in descent order
    w, h = self.size(channel = 'depth');
    proj = self.project((w, h), v[s]); # proj.shape = (h * w, 2)
    j, i = proj.astype(np.uint32).T; # get u, v of homogeneous coordinate, shape = (h * w)
    total_cam_id = total_cam_id[s]; # reorder cam id with the sorted sequence, shape = (h * w)
    # mask of visible voxel in image area
    im = (i >= 0) & (i < h);
    jm = (j >= 0) & (j < w);
    m = im & jm;
    cw, ch = self.size(channel = 'color');
    # turn the texcoordinate into absolute coordinate
    v, u = (total_texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T;
    # clip texcoordinate within captured image area
    np.clip(u, 0, ch - 1, out = u);
    np.clip(v, 0, cw - 1, out = v);
    # output
    out = np.empty((h, w, 3), dtype = np.uint8);
    out[i[m], j[m]] = total_colors[total_cam_id[m], u[m], v[m]];
    return out;

  def size(self, cam_id = 0, channel = 'depth'):

    return IMG_WIDTH, IMG_HEIGHT;

  def change_distance(self, dz):

    self.distance -= dz;
    # south: increase on z axis
    self.translations[0][2] += dz;
    # east: increase on x axis
    self.translations[1][2] += dz;
    # north: decrease on z axis
    self.translations[2][2] += dz;
    # west: descrease on x axis
    self.translations[3][2] += dz;

  def change_translation(self, dp):

    # south
    Rx, _ = cv2.Rodrigues((self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw, 0));
    rotation = np.dot(Ry, Rx).astype(np.float32);
    self.translations[0] -= np.dot(rotation, dp, dtype = np.float32);
    # east
    Rx, _ = cv2.Rodrigues((0, 0, -self.pitch));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2, 0));
    rotation = np.dot(Ry, Rx).astype(np.float32);
    self.translations[1] -= np.dot(rotation, dp, dtype = np.float32);
    # north
    Rx, _ = cv2.Rodrigues((-self.pitch, 0, 0));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2 * 2, 0));
    rotation = np.dot(Ry, Rx).astype(np.float32);
    self.translations[2] -= np.dot(rotation, dp, dtype = np.float32);
    # west
    Rx, _ = cv2.Rodrigues((0, 0, self.pitch));
    Ry, _ = cv2.Rodrigues((0, self.yaw + pi / 2 * 3, 0));
    rotation = np.dot(Ry, Rx).astype(np.float32);
    self.translations[3] -= np.dot(rotation, dp, dtype = np.float32);

  def change_euler(self, dyaw, dpitch):

    self.yaw += dyaw;
    self.pitch += dpitch;
    
  def reset_position(self):

    self.distance = 2;
    self.translations[0] = np.array([0, 0, -1], dtype = np.float32); # south
    self.translations[1] = np.array([-2, 0, 1], dtype = np.float32); # east
    self.translations[2] = np.array([0, 0, 3], dtype = np.float32); # north
    self.translations[3] = np.array([2, 0, 1], dtype = np.float32); # west
    self.pitch = 0;
    self.yaw = 0;

if __name__ == "__main__":

  va = VisualizeAdjustment();
  prev_mouse = (0, 0);
  w, h = va.size();
  mouse_btns = [False, False, False];

  def mouse_cb(event, x, y, flags, param):

    global prev_mouse;
    if event == cv2.EVENT_LBUTTONDOWN:

      mouse_btns[0] = True;

    if event == cv2.EVENT_LBUTTONUP:

      mouse_btns[0] = False;

    if event == cv2.EVENT_RBUTTONDOWN:

      mouse_btns[1] = True;

    if event == cv2.EVENT_RBUTTONUP:

      mouse_btns[1] = False;

    if event == cv2.EVENT_MBUTTONDOWN:

      mouse_btns[2] = True;

    if event == cv2.EVENT_MBUTTONUP:

      mouse_btns[2] = False;

    if event == cv2.EVENT_MOUSEMOVE:

      dx, dy = x - prev_mouse[0], y - prev_mouse[1];

      if mouse_btns[0]:

        va.change_euler(float(dx) / w * 2, -float(dy) / h * 2);

      if mouse_btns[1]:

        dp = np.array((dx / w, dy / h, 0), dtype = np.float32)
        va.change_translation(dp);

      if mouse_btns[2]:
        
        dz = sqrt(dx**2 + dy**2) + copysign(0.01, -dy);
        va.change_distance(dz);

    prev_mouse = (x, y);

  cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE);
  cv2.resizeWindow('show', w, h);
  cv2.setMouseCallback('show', mouse_cb);

  while True:
    image = va.visualize();
    cv2.imshow('show', image);
    key = cv2.waitKey(1);
    if key == ord('q'):
      print('calibrated optimal distance is %d' % distance);
      break;
    if key == ord('r'):
      va.reset_position();
