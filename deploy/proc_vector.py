import face_model
import argparse
import os
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--enable-gpu', default=False, type=bool, help='enable to detect and inference in GPU')
#parser.add_argument('--model', default='../models2/model-r100-sfz/model,8', help='path to load model.')
parser.add_argument('--model', default='../../model/m1-insightv3/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--mode', default=1, type=int, help='mode, 0: do nothing, 1:append or merge depends on merge threshold')
parser.add_argument('--threshold', default=0.4, type=float, help='cosine threshold')
parser.add_argument('--merge-threshold', default=0.8, type=float, help='cosine threshold for merge. 1.0: never do merging; -1.0: to merge everytime')
parser.add_argument('--append-threshold', default=0.55, type=float, help='cosine threshold for append. 1.0: never do append; -1.0: to append everytime')
parser.add_argument('--only-replace', default=False, type=bool, help='replace most similar vecotor when list is full,no merge')
parser.add_argument('--max_vector_size', default=8, type=int, help='')
parser.add_argument('--input', default='camera-video1', type=str, help='input npy file name')
args = parser.parse_args()

merge_count=0
append_count=0
full_count=0
none_count=0
print 'count init!'

X = np.load(args.input+'/X.npy')

#for i in xrange(X.shape[0]):
#  if i==0:
#    continue
#  a = X[i]
#  sims = []
#  for j in xrange(0, i):
#    b = X[j]
#    sim = np.dot(a, b)
#    sims.append(sim)
#  print(i,max(sims))

def update_sim_score(vec_list):
  for i,vec1 in enumerate(vec_list):
    max_ids  = 0
    max_sims = 0.0
    for j,vec2 in enumerate(vec_list):
      if i == j:
        next
      else:
        sim = np.dot(vec1[0], vec2[0])
        if sim > max_sims:
          max_sims = sim
          max_ids = j
    vec1[1] = max_ids
    vec1[2] = max_sims

def insert_vec(vec_list, input_vec, input_id, input_score):
  global merge_count
  global append_count
  global full_count
  global none_count
  if input_score > args.merge_threshold:
    print 'do merge'
    new_vec = vec_list[input_id][0] + input_vec
    new_vec = new_vec / np.linalg.norm(new_vec)
    vec_list[input_id][0] = new_vec
    update_sim_score(vec_list)
    merge_count+=1
  elif input_score < args.append_threshold:
    if len(vec_list) < args.max_vector_size:
      vec_list.append([input_vec,input_id,input_score])
      update_sim_score(vec_list)
      print 'append to list'
      append_count+=1
    else:
      max_score = 0.0
      max_id = -1
      print 'merge when list if full'
      full_count+=1
      for i,vec in enumerate(vec_list):
        if max_score < vec[2]:
          max_score = vec[2]
          max_id = i
      if max_score > input_score:
        if args.only_replace==False:
          vec2_id = vec_list[max_id][1]
          new_vec = vec_list[max_id][0] + vec_list[vec2_id][0]
          new_vec = new_vec / np.linalg.norm(new_vec)
          vec_list[max_id][0]  = new_vec
          vec_list[vec2_id][0] = input_vec
          update_sim_score(vec_list)
        else:
          vec_list[max_id][0] = input_vec
          update_sim_score(vec_list)
      else:
        if args.only_replace==False:
          new_vec = input_vec + vec_list[input_id][0]
          new_vec = new_vec / np.linalg.norm(new_vec)
          vec_list[input_id][0]  = new_vec
          update_sim_score(vec_list)
        else:
          pass
  else:
    none_count+=1

identities = {0: [[X[0],0,0]]}
face_img = {0:[0]}
for i in xrange(1, X.shape[0]):
  a = X[i]
  ids = []
  sims = []
  vector_indexes = []
  for _id, vectors in identities.iteritems():
    for vector_idx, vector in enumerate(vectors):
      sim = np.dot(a, vector[0])
      sims.append(sim)
      ids.append(_id)
      vector_indexes.append(vector_idx)
  max_idx = np.argmax(sims)
  max_score = sims[max_idx]
  max_id = ids[max_idx]
  max_vector_idx  = vector_indexes[max_idx]
  print(i, max_score, max_id)
  if max_score<args.threshold:
    new_id = len(identities)
    identities[new_id] = [[a,0,0]]
    face_img[new_id] = [i]
    print 'append new id ', new_id
  else:
    if args.mode==1:
      face_img[max_id].append(i)
      insert_vec(identities[max_id],a,max_vector_idx,max_score)

print('final', len(identities))
print 'merge_count ', merge_count
print 'append_count ', append_count
print 'full_count', full_count
print 'none_count', none_count

os.system('mkdir ' + args.input + '/id')

for face_id,img_list in face_img.items():
  new_dir = args.input + '/id/' + str(face_id)
  os.system('mkdir ' + new_dir)
  for img in img_list:
    src = args.input + '/' + str(img) + '.jpg '
    os.system('cp ' + src + new_dir)
    print('face_id:%d, img idx:%d' % (face_id,img))