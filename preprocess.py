#!/usr/bin/python3

import sys;
import numpy as np;
import cv2;

def intersection(a,b):

    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0: return (0,0,0,0);
    return (x, y, w, h);

def union(a,b):

    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def preprocess(image):
    
    ret, mask = cv2.threshold(image, 1,255,cv2.THRESH_BINARY);
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S);
    comp_boundings = dict();
    for label in range(num_labels):
        comp_mask = np.where(labels == label);
        # skip small component
        if len(comp_mask[0]) < 100 or len(comp_mask[0]) > 1000000: continue;
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
            cv2.rectangle(image, tuple(ul),tuple(dr), (255,255,255), 2);
    cv2.imshow('comp', image);
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
