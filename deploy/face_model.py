from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    if args.enable_gpu == True:
      ctx = mx.gpu(args.gpu)
    else:
      ctx = mx.cpu()
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    #self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.65,0.75,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox_list, points_list = ret
    if bbox_list.shape[0]==0:
      return None
    
    aligned_list = []
    nimg_list = []
    pose_type_list = []

    for bbox, points in zip(bbox_list, points_list):
      bbox = bbox[0:4]
      points = points.reshape((2,5)).T
      #print(bbox)
      #print(points)
      nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
      nimg_list.append(nimg)
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      aligned_list.append(aligned)
      pose_type,left_score,right_score,_,_ = self.check_large_pose(points,bbox)
      pose_type_list.append((pose_type,left_score,right_score))

    return aligned_list, nimg_list, pose_type_list

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

  def check_large_pose(self, landmark, bbox):
      assert landmark.shape==(5,2)
      assert len(bbox)==4
      def get_theta(base, x, y):
        vx = x-base
        vy = y-base
        vx[1] *= -1
        vy[1] *= -1
        tx = np.arctan2(vx[1], vx[0])
        ty = np.arctan2(vy[1], vy[0])
        d = ty-tx
        d = np.degrees(d)
        #print(vx, tx, vy, ty, d)
        #if d<-1.*math.pi:
        #  d+=2*math.pi
        #elif d>math.pi:
        #  d-=2*math.pi
        if d<-180.0:
          d+=360.
        elif d>180.0:
          d-=360.0
        return d
      landmark = landmark.astype(np.float32)

      theta1 = get_theta(landmark[0], landmark[3], landmark[2])
      theta2 = get_theta(landmark[1], landmark[2], landmark[4])
      #print(va, vb, theta2)
      theta3 = get_theta(landmark[0], landmark[2], landmark[1])
      theta4 = get_theta(landmark[1], landmark[0], landmark[2])
      theta5 = get_theta(landmark[3], landmark[4], landmark[2])
      theta6 = get_theta(landmark[4], landmark[2], landmark[3])
      theta7 = get_theta(landmark[3], landmark[2], landmark[0])
      theta8 = get_theta(landmark[4], landmark[1], landmark[2])
      #print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
      left_score = 0.0
      right_score = 0.0
      up_score = 0.0
      down_score = 0.0
      if theta1<=0.0:
        left_score = 10.0
      elif theta2<=0.0:
        right_score = 10.0
      else:
        left_score = theta2/theta1
        right_score = theta1/theta2
      if theta3<=10.0 or theta4<=10.0:
        up_score = 10.0
      else:
        up_score = max(theta1/theta3, theta2/theta4)
      if theta5<=10.0 or theta6<=10.0:
        down_score = 10.0
      else:
        down_score = max(theta7/theta5, theta8/theta6)
      mleft = (landmark[0][0]+landmark[3][0])/2
      mright = (landmark[1][0]+landmark[4][0])/2
      box_center = ( (bbox[0]+bbox[2])/2,  (bbox[1]+bbox[3])/2 )
      ret = 0
      threshold = [5.0,4.0]
      if left_score>=threshold[0]:
        ret = 1
      if ret==0 and left_score>=threshold[1]:
        if mright<=box_center[0]:
          ret = 1
      if ret==0 and right_score>=threshold[0]:
        ret = 2
      if ret==0 and right_score>=threshold[1]:
        if mleft>=box_center[0]:
          ret = 2
      if ret==0 and up_score>=2.0:
        ret = 3
      if ret==0 and down_score>=5.0:
        ret = 4
      return ret, left_score, right_score, up_score, down_score