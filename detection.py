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
from PIL import Image
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
import Signboard_Detector.roi_helpers as roi_helpers
from Signboard_Detector import test_map
def detect(model_path,config_output_filename,filepath,record_path):
    st = time.time()
    # Load the Config file
    with open(config_output_filename, 'rb') as f_in:
      C = pickle.load(f_in)
    
    # Maximum Boxes for RPN Proposals
    Max_boxes = 300
    # Number of region for RPN Proposal
    num_proposal_region = 256
    #applying non-maximum suppression we can prune the number of overlapping bounding boxes down to one.
    # Overlap Threshold for nms of rpn to roi
    rpn_ov_thresh = 0.9
    # Overlap Threshold for nms of testing
    nms_ov_thresh = 0.2
    # Training set size
    train_size = 4000 # 90% 0f 2600 data
    # Image input shape for model input
    img_in_shape = (None, None, 3) # RGB -> (None, None, 3) 
    # Bounding Box threshold for testing classifier model bboxes filtering
    bbox_threshold = 0.7
    network = 'vgg16'
    if network == 'vgg16':
      C.network = 'vgg16'
      from Signboard_Detector import vgg16 as nn
    elif network == 'resnet50':
      C.network = 'resnet50'
      from Signboard_Detector import resnet50 as nn
    elif network == 'inceptionV4':	
      C.network = 'inceptionV4'
      from Signboard_Detector import inceptionV4 as nn
    elif network == 'densenet121':
      C.network = 'densenet121'   
      from Signboard_Detector import densenet121 as nn
    else:
      print('Not a valid model')
      raise ValueError
    record_df = pd.DataFrame(columns=['Label_Name', 'X1','Y1','X2','Y2','Width','Height'])

    num_features = 512

    input_shape_img_ = img_in_shape
    input_shape_features_ = (None, None, num_features)

    img_input_ = Input(shape=input_shape_img_)
    roi_input_ = Input(shape=(C.num_rois, 4))
    feature_map_input_ = Input(shape=input_shape_features_)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers_ = nn.nn_base(img_input_, trainable=True)

    # define the RPN, built on the base layers
    num_anchors_ = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers_ = nn.rpn(shared_layers_, num_anchors_)
    classifier_ = nn.classifier(feature_map_input_, roi_input_, C.num_rois, nb_classes=len(C.class_mapping))
    model_rpn_ = Model(img_input_, rpn_layers_)
    model_classifier_only_ = Model([feature_map_input_, roi_input_], classifier_)
    model_classifier_ = Model([feature_map_input_, roi_input_], classifier_)

    print('Loading weights from {}'.format(model_path))
    model_rpn_.load_weights(model_path, by_name=True)
    model_classifier_.load_weights(model_path, by_name=True)

    model_rpn_.compile(optimizer='sgd', loss='mse')
    model_classifier_.compile(optimizer='sgd', loss='mse')

    # Switch key value for class mapping
    class_mapping_ = C.class_mapping
    class_mapping_= {v: k for k, v in class_mapping_.items()}
    #print('class map: ',class_mapping_)
    class_to_color = {class_mapping_[v]: np.random.randint(0, 255, 3) for v in class_mapping_}

    # If the box classification value is less than this, we ignore this box

    img = cv2.imread(filepath)
    #img = Image.open(filepath)
    wid, hei, _ = img.shape
    if wid !=1000 and hei !=600:
      #img = img.resize((1000, 600), Image.ANTIALIAS)
      img = cv2.resize(img, (1000, 600), interpolation=cv2.INTER_CUBIC)
      print(img.shape)


    X, ratio,fx,fy = test_map.format_img(img, C)
    # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
    X = np.transpose(X, (0, 2, 3, 1))
    
    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn_.predict(X)

    # Get bboxes by applying NMS 
    # R.shape = (300, 4)
    # higer overlap_thresh performs lighter elimination and lower overlap_thresh performs strict elimination on RPN Bboxes
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), max_boxes=Max_boxes,overlap_thresh=rpn_ov_thresh)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only_.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        # Drop 'bg' classes bboxes
        for ii in range(P_cls.shape[1]):

            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
              continue
              
            # Get class name
            cls_name = class_mapping_[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])
        # Apply non-max-suppression on final bboxes to get the output bounding boxe
        # higer overlap_thresh performs lighter elimination and lower overlap_thresh performs strict elimination on RPN Bboxes
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), 
                                                                    overlap_thresh=nms_ov_thresh,max_boxes=Max_boxes)

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = test_map.get_real_coordinates(ratio, x1, y1, x2, y2)
            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)
            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            
            new_row = {'Label_Name':textLabel, 
									'X1':real_x1, 
									'Y1':real_y1, 
									'X2':real_x2, 
									'Y2':real_y2, 
									'Width':abs(real_x2-real_x1), 
									'Height':abs(real_y2-real_y1) 
									 }
            record_df = record_df.append(new_row, ignore_index=True)
            record_df.to_csv(record_path, index=0)

    print('Elapsed time [Sec] = {}'.format(time.time() - st))
    
    plt.figure(figsize=(15,15))
    plt.grid()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imsave("result.jpg",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))