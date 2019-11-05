#!/usr/bin/python3

import sys;
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

def preprocess(image):

    # 1) find a foot print instance (bounding, pixels)
    ret, mask = cv2.threshold(image, 1,255,cv2.THRESH_BINARY);
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S);
    comp_boundings = dict();
    for label in range(num_labels):
        comp_mask = np.where(labels == label);
        # skip small component
        if len(comp_mask[0]) < 80 or len(comp_mask[0]) > 1000000: continue;
        # comp_coor.shape = (number, 2)
        comp_coor = np.array(list(zip(comp_mask[1],comp_mask[0])));
        x,y,w,h = cv2.boundingRect(comp_coor);
        # swelled version of the bounding
        center = np.array([x + w // 2, y + h // 2]);
        swelled = (center[0] - 2 * w, center[1] - h // 2, 4 * w, h);
        merged = False;
        for key, comp in comp_boundings.items():
            intersect = intersection(comp[1], swelled);
            if intersect[2] * intersect[3] > 0:
                # merge to existing component is they are intersected
                comp_boundings[key] = (
                    np.concatenate([comp[0], comp_coor], axis = 0),
                    union(comp[1], (x, y, w, h))
                );
                merged = True;
                break;
        if False == merged:
            comp_boundings[label] = (comp_coor, (x,y,w,h));

    if True:
        for key, comp in comp_boundings.items():
            ul = np.array(comp[1][0:2]);
            dr = np.array(comp[1][0:2]) + np.array(comp[1][2:4]);
            cv2.rectangle(mask, tuple(ul),tuple(dr), (255,255,255), 2);

    # 2) calculate mean and eigvec of each foot print
    for key, foot in comp_boundings.items():
        pts = foot[0];
        bounding = foot[1];
        # center.shape = (2,)
        center = mean(pts, image, True);
        # cov.shape = (2,2)
        cov = covariance(pts, center, image, True);
        e, v = eigvec(cov);
        ref = tf.math.atan2(v[1,1],v[0,1]);
        # polar mean
        dist, angle = polar_mean(pts, center);
        diff = angle - ref;
        if True:
            # draw center of the foot
            cv2.circle(mask, tuple(center.astype('int32')), 5, (255,255,255));
            # mean direction
            pts1 = center + 0.4 * e[0] * v[:,0];
            pts2 = center + 0.4 * e[1] * v[:,1];
            cv2.line(mask, tuple(center.astype('int32')), tuple(pts1.astype('int32')), (255,255,255), 1);
            cv2.line(mask, tuple(center.astype('int32')), tuple(pts2.astype('int32')), (255,255,255), 1);
            # print mean dist and mean polar
            cv2.putText(mask, str(angle - ref.numpy()), (bounding[0],bounding[1]-4), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255), 1, 8);

    cv2.imshow('comp', mask);
    cv2.waitKey();

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <image>");
        exit(1);
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE);
    if img is None:
        print("invalid image!");
        exit(1);
    preprocess(img);
