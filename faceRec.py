# run file to train the program to recognize the image
# the more you run it, the more acurate the program detects the image
import cv2
import os
from PIL import Image
import numpy as np
import pickle

# Haar Cascade function: Face detection of the image
# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face_LBPHFaceRecognizer.create()


# retrieve the directory of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# retrieve image directory from folder images/...
image_dir = os.path.join(BASE_DIR, "images")

cur_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    
    for file in files:

        if file.endswith("jpg"):
           
           # retrieve path of image
            path = os.path.join(root, file)

            # retrieve label name
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            
            # print(label,path)

            if not label in label_ids:
                label_ids[label] = cur_id
                cur_id += 1 
            
            id = label_ids[label]
            
            #print(label_ids)

            # y_labels.append(label)
            # x_train.append(path)

            # convert image into gray scale
            pil_img = Image.open(path).convert("L")
            size = (550,550)
            final_img  = pil_img.resize(size, Image.Resampling.LANCZOS)

            # append to numpy.array
            img_arr = np.array(final_img,"uint8")
            # print(img_arr)

            # detect face
            faces = faceCascade.detectMultiScale(
                img_arr,
                scaleFactor = 1.5,
                minNeighbors = 5
            )

            for x, y, w, h in faces:
                roi = img_arr[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)

# print(y_labels)
# print(x_train)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")