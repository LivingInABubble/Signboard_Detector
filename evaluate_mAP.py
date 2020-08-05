from __future__ import division
import random
import pprint
import sys
import cv2
import os
import time
import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from Signboard_Detector.simple_parser import get_data
from Signboard_Detector import config, data_generators
from Signboard_Detector import losses 
from Signboard_Detector import test_map
import Signboard_Detector.roi_helpers as roi_helpers


def calc_mAP(test_imgs,model_rpn_,model_classifier_only_,C,class_mapping_,Max_boxes,rpn_ov_thresh,nms_ov_thresh):
  T = {}
  P = {}
  mAPs = []
  for idx, img_data in enumerate(test_imgs):
      #print('{}/{}'.format(idx,len(test_imgs)))
      st = time.time()
      filepath = img_data['filepath']
      img = cv2.imread(filepath)
      X, ratio ,fx, fy = test_map.format_img(img, C)
      # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
      X = np.transpose(X, (0, 2, 3, 1))
      # get the feature maps and output from the RPN
      [Y1, Y2, F] = model_rpn_.predict(X)
      R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(),max_boxes=Max_boxes,overlap_thresh=rpn_ov_thresh)
      # convert from (x1,y1,x2,y2) to (x,y,w,h)
      R[:, 2] -= R[:, 0]
      R[:, 3] -= R[:, 1]
      # apply the spatial pyramid pooling to the proposed regions
      bboxes = {}
      probs = {}
      for jk in range(R.shape[0] // C.num_rois + 1):
          ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
          if ROIs.shape[1] == 0:
              break

          if jk == R.shape[0] // C.num_rois:
              # pad R
              curr_shape = ROIs.shape
              target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
              ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
              ROIs_padded[:, :curr_shape[1], :] = ROIs
              ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
              ROIs = ROIs_padded

          [P_cls, P_regr] = model_classifier_only_.predict([F, ROIs])

          # Calculate all classes' bboxes coordinates on resized image (300, 400)
          # Drop 'bg' classes bboxes
          for ii in range(P_cls.shape[1]):

              # If class name is 'bg', continue
              if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                  continue

              # Get class name
              cls_name = class_mapping_[np.argmax(P_cls[0, ii, :])]

              if cls_name not in bboxes:
                  bboxes[cls_name] = []
                  probs[cls_name] = []

              (x, y, w, h) = ROIs[0, ii, :]

              cls_num = np.argmax(P_cls[0, ii, :])
              try:
                  (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                  tx /= C.classifier_regr_std[0]
                  ty /= C.classifier_regr_std[1]
                  tw /= C.classifier_regr_std[2]
                  th /= C.classifier_regr_std[3]
                  x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
              except:
                  pass
              bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
              probs[cls_name].append(np.max(P_cls[0, ii, :]))

      all_dets = []

      for key in bboxes:
          bbox = np.array(bboxes[key])

          # Apply non-max-suppression on final bboxes to get the output bounding boxe
          new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=nms_ov_thresh,max_boxes=Max_boxes)
          for jk in range(new_boxes.shape[0]):
              (x1, y1, x2, y2) = new_boxes[jk, :]
              det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
              all_dets.append(det)


      #print('Elapsed time = {}'.format(time.time() - st))
      t, p = test_map.get_map(all_dets, img_data['bboxes'], (fx, fy))
      for key in t.keys():
          if key not in T:
              T[key] = []
              P[key] = []
          T[key].extend(t[key])
          P[key].extend(p[key])
      all_aps = []
      for key in T.keys():
          ap = average_precision_score(T[key], P[key])
          #print('{} AP: {}'.format(key, ap))
          all_aps.append(ap)
      print('mAP = {}'.format(np.mean(np.array(all_aps))))
      
      mAPs.append(np.mean(np.array(all_aps)))
      #print(T)
      #print(P)
  return mAPs
