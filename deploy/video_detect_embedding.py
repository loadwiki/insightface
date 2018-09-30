import face_model
import argparse
import os
import cv2
import sys
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--enable-gpu', default=False, type=bool, help='enable to detect and inference in GPU')
#parser.add_argument('--model', default='../models2/model-r100-sfz/model,8', help='path to load model.')
parser.add_argument('--model', default='../../../model/model-r100-gg/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--input-video', default='/Users/load/code/cpp/tracking/video/camera-244-crop-8p.mov', type=str, help='input video file')
parser.add_argument('--detect-peroid', default=20, type=int, help='detect face per frames')
parser.add_argument('--output', default='camera-244', type=str, help='output npy file name')
args = parser.parse_args()

o_dir = args.output + '/'
mkdir_cmd = 'mkdir -p ' + args.output
os.system('rm -rf '+args.output)
os.system(mkdir_cmd)
os.system('mkdir -p ' + args.output + '/original')
for i in range(5):
  os.system('mkdir -p ' + args.output + '/' + str(i))

model = face_model.FaceModel(args)

cap = cv2.VideoCapture(args.input_video)

ret = True
frame_count = 0
X = []

while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == False:
    break
  frame_count += 1
  print 'frame_count ',frame_count
  if frame_count % args.detect_peroid != 0:
    continue
  
  cv2.imwrite(args.output + '/original/' + str(frame_count) + '.jpg', frame)

  face_ret = model.get_input(frame)
  if face_ret is None:
    print('face not found')
    continue
  aligned_list, nimg_list, pose_type_list = face_ret

  idx = 0
  for aligned, nimg, pose_type in zip(aligned_list, nimg_list, pose_type_list):
    cv2.imwrite(args.output + '/' + str(pose_type) + '/' + str(frame_count) + '_' + str(idx) + '.jpg', nimg)
    idx += 1
    x = model.get_feature(aligned)
    X.append(x)
  #break
  #cv2.imshow('face',crop)
  #cv2.waitKey(5)
  
X = np.array(X)
np.save(args.output+'/X',X)
cap.release()
# When everything done, release the capture
