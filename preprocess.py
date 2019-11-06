#!/usr/bin/python3

import sys;
from math import cos, sin, atan2;
import numpy as np;
import tensorflow as tf;
import cv2;

def intersection(a,b):

    x = max(a[0], b[0]);
    y = max(a[1], b[1]);
    w = min(a[0]+a[2], b[0]+b[2]) - x;
    h = min(a[1]+a[3], b[1]+b[3]) - y;
    if w < 0 or h < 0: return (0,0,0,0);
    return (x, y, w, h);

def union(a,b):

    x = min(a[0], b[0]);
    y = min(a[1], b[1]);
    w = max(a[0]+a[2], b[0]+b[2]) - x;
    h = max(a[1]+a[3], b[1]+b[3]) - y;
    return (x, y, w, h);

def mean(pts, img = None, weighted = False):
    
    assert type(pts) is np.ndarray and len(pts.shape) == 2 and pts.shape[1] == 2;
    if weighted == True: assert img is not None;
    if weighted:
        img = tf.cast(img, dtype = tf.float32);
        weights = tf.gather_nd(img, tf.reverse(pts, axis = [1]));
        weights = weights / tf.math.reduce_sum(weights);
        pts = tf.constant(pts, dtype = tf.float32) * tf.expand_dims(weights, axis = 1);
        mean = tf.math.reduce_sum(pts, axis = 0);
    else:
        pts = tf.constant(pts, dtype = tf.float32);
        mean = tf.math.reduce_mean(pts, axis = 0);
    return mean.numpy();

def polar_mean(pts, center):
    
    assert type(pts) is np.ndarray and len(pts.shape) == 2 and pts.shape[1] == 2;
    assert type(center) is np.ndarray and len(center.shape) == 1 and center.shape[0] == 2;
    # dev.shape = (n, 2)
    dev = pts - center;
    # distances.shape = (n,)
    distances = tf.norm(dev, axis = 1);
    mask = tf.math.less(distances, 20);
    # angles.shape =(n,)
    dev = tf.boolean_mask(dev, mask);
    angles = tf.math.atan2(dev[:,1], dev[:,0]);
    mean_dist = tf.math.reduce_mean(distances);
    mean_angle = tf.math.reduce_mean(angles);
    return mean_dist.numpy(), mean_angle.numpy();

def covariance(pts, center = None, img = None, weighted = False):

    assert type(pts) is np.ndarray and len(pts.shape) == 2 and pts.shape[1] == 2;
    if weighted == True: assert img is not None;
    x = tf.cast(pts, dtype = tf.float32);
    if center is None:
        # centralized.shape = (n, 2)
        centralized = x - mean(x);
    else:
        centralized = x - center;
    if weighted:
        img = tf.cast(img, dtype = tf.float32);
        weights = tf.gather_nd(img, tf.reverse(pts, axis = [1]));
        weights = weights / tf.math.reduce_sum(weights);
        diag = tf.linalg.diag(weights);
        cov = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(centralized, (1,0)), diag), centralized);
        cov = cov / (1 - tf.math.reduce_sum(tf.math.pow(weights,2)));
    else:
        # cov.shape = (2,2)
        cov = tf.linalg.matmul(tf.transpose(centralized, (1, 0)), centralized) / (pts.shape[0] - 1);
    return cov.numpy();

def eigvec(cov):

    assert type(cov) is np.ndarray and len(cov.shape) == 2 and cov.shape[1] == 2;
    cov = tf.constant(cov, dtype = tf.float32);
    e,v = tf.linalg.eigh(cov);
    return e.numpy(), v.numpy();

def crop(image, center, angle, src_size, dst_size):
    
    img = image.copy();
    # translate image to make center at origin position
    # anti-clockwise rotate by angle
    # translate image to make the origin at the center position
    translate1 = np.eye(3, dtype = np.float32);
    translate1[0,2] = -center[0];
    translate1[1,2] = -center[1];
    rotate = np.eye(3, dtype = np.float32);
    rotate[0,0] = cos(-angle);  rotate[0,1] = -sin(-angle);
    rotate[1,0] = sin(-angle);  rotate[1,1] = cos(-angle);
    translate2 = np.eye(3, dtype = np.float32);
    translate2[0,2] = -center[0];
    translate2[1,2] = -center[1];
    affine = np.dot(translate2,np.dot(rotate,translate1));
    affine = affine[0:2,:];
    return cv2.warpAffine(img, affine, dst_size);

def preprocess(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    else:
        gray = image.copy();
    # 1) find a foot print instance (bounding, pixels)
    ret, mask = cv2.threshold(gray, 1,255,cv2.THRESH_BINARY);
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S);
    comp_boundings = dict();
    for label in range(num_labels):
        comp_mask = np.where(labels == label);
        # skip small component
        if len(comp_mask[0]) < 50 or len(comp_mask[0]) > 1000000: continue;
        # comp_coor.shape = (number, 2)
        comp_coor = np.array(list(zip(comp_mask[1],comp_mask[0])));
        x, y, w, h = cv2.boundingRect(comp_coor);
        while True:
            # swelled version of the bounding
            center = np.array([x + w // 2, y + h // 2]);
            swelled = (center[0] - 2 * w, center[1] - h // 2, 4 * w, h);
            merge_list = list();
            for key, comp in comp_boundings.items():
                intersect = intersection(comp[1], swelled);
                if intersect[2] * intersect[3] > 0:
                    merge_list.append(key);
            if 0 == len(merge_list): break;
            for key in merge_list:
                comp_coor = np.concatenate([comp_boundings[key][0], comp_coor], axis = 0);
                x, y, w, h = union(comp_boundings[key][1], (x, y, w, h));
                del comp_boundings[key];
        comp_boundings[label] = (comp_coor, (x, y, w, h));

    if True:
        for key, comp in comp_boundings.items():
            ul = np.array(comp[1][0:2]);
            dr = np.array(comp[1][0:2]) + np.array(comp[1][2:4]);
            cv2.rectangle(image, tuple(ul),tuple(dr), (255,255,255), 2);

    # 2) calculate mean and eigvec of each foot print
    leftfeet = list();
    rightfeet = list();
    for key, foot in comp_boundings.items():
        pts = foot[0];
        bounding = foot[1];
        # center.shape = (2,)
        center = mean(pts, gray, True);
        # cov.shape = (2,2)
        cov = covariance(pts, center, gray, True);
        e, v = eigvec(cov);
        ref = atan2(v[1,1],v[0,1]);
        # polar mean
        dist, angle = polar_mean(pts, center);
        diff = angle - ref;
        # eigenvector angle
        angle = atan2(v[1,1], v[0,1]);
        length = tf.norm(0.18 * e[1] * v[:,1]).numpy();
        foot = crop(gray, center, angle, (length, length), (250, 250));
        cv2.imshow('foot', foot);
        #cv2.waitKey();
        if diff < 0:
            # left foot
            pass;
        else:
            # right foot
            pass;
        if True:
            # draw center of the foot
            cv2.circle(image, tuple(center.astype('int32')), 5, (255,255,255));
            # mean direction
            pts1 = center + 0.09 * e[0] * v[:,0];
            pts2 = center + 0.09 * e[1] * v[:,1];
            cv2.line(image, tuple(center.astype('int32')), tuple(pts1.astype('int32')), (255,255,255), 1);
            cv2.line(image, tuple(center.astype('int32')), tuple(pts2.astype('int32')), (255,255,255), 1);
            # print mean dist and mean polar
            cv2.putText(image, str(diff), (bounding[0],bounding[1]-4), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255), 1, 8);

    cv2.imshow('comp', image);
    cv2.waitKey();

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <image>");
        exit(1);
    img = cv2.imread(sys.argv[1]);
    if img is None:
        print("invalid image!");
        exit(1);
    preprocess(img);
