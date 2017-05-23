import cv2
import numpy as np
import cPickle as pickle

def pickle_keypoints(keypoints, descriptors, img):
  i = 0
  array = []
  array.append(img)
  temp_array = []
  for point in keypoints:
    temp = (point.pt, point.size, point.angle, point.response, point.octave,
    point.class_id, descriptors[i])
    ++i
    temp_array.append(temp)
  array.append(temp_array)
  return array

def unpickle_keypoints(array):
  keypoints = []
  descriptors = []
  img = array[0]
  for point in array[1]:
    temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1],
      _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    temp_descriptor = point[6]
    keypoints.append(temp_feature)
    descriptors.append(temp_descriptor)
  return keypoints, np.array(descriptors), img
