import cv2
import numpy as np
import glob
from main import *

# get all image_url in database
database_images = glob.glob("./database/*.jpg")

# Load each image
for image_url in database_images:
  number = image_url.split("/")[2].split(".")[0]
  img = cv2.imread(image_url)
  print(number)
  # Convert them to greyscale
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # sift extraction
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(gray,None)

  #Store and Retrieve keypoint features
  temp = pickle_keypoints(kp, des, image_url)
  file = "./sift_detector_database/" + number + ".txt"
  pickle.dump(temp, open(file, "wb"))
