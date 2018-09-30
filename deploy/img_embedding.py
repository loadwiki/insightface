import face_model
import argparse
import os
import cv2
import sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
#parser.add_argument('--model', default='../models2/model-r100-sfz/model,8', help='path to load model.')
parser.add_argument('--model', default='../../model/model-r100-gg/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--input-dir', default='./test', type=str, help='input image dir')
parser.add_argument('--output', default='feature_vectors', type=str, help='input image dir')

args = parser.parse_args()

model = face_model.FaceModel(args)
#args.input_dir = 'test'
image_files = os.listdir(args.input_dir)

#image_files = sorted(image_files, key = lambda x : int(x.split('.')[0].split('_')[1]))
#df = pd.DataFrame(image_files)
#df.to_csv('img_list')

X = []
for f in image_files:
  print(f)
  file = os.path.join(args.input_dir, f)
  img = cv2.imread(file)
  img = np.transpose(img, (2,0,1))
  x = model.get_feature(img)
  X.append(x)

X = np.array(X)

np.save(args.output,X)
