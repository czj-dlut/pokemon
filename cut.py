import os
import cv2
file_dir = "C:/Users/Auror/Desktop/pokemon/data/test"

for root, dirs, files in os.walk(file_dir):  
    for file in files: 
        img = cv2.imread(root+"/"+file,1)
        cv2.imwrite(root+"/"+file,img)