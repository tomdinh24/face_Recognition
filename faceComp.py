import cv2  
import pickle
import numpy as np

def faceComp(image_path1, image_path2):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("trainner.yml")

    labels = {}

    with open("labels.pickle", "rb") as f:
        oldlabels = pickle.load(f)
        labels = {v:k for k,v in oldlabels.items()}

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)


    # capture by frame
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # detect face
    faces1 = faceCascade.detectMultiScale( gray1, scaleFactor = 1.5, minNeighbors = 5, minSize = (30,30))
    faces2 = faceCascade.detectMultiScale( gray2, scaleFactor = 1.5, minNeighbors = 5, minSize = (30,30))

    # box around face
    for x,y,w,h in faces1:
        roi_gray1 = gray1[y:y+h, x:x+w]

    for x,y,w,h in faces2:
        roi_gray2 = gray2[y:y+h, x:x+w]

    # retrieve user id and confidence level
    id_1, conf_1 = recognizer.predict(roi_gray1)
    print("face1: ", labels[id_1], " - confident level: ",conf_1 )

    id_2, conf_2 = recognizer.predict(roi_gray2)
    print("face2: ", labels[id_2], " - confident level: ",conf_2 )

    # if confidence level is high then it is a match
    # if confidence levle is low then it can't predict the user
    if conf_1 >= 70 and conf_2 >= 70:
        if id_1 == id_2:
            print("MATCH of \"{}\" type photos!".format(labels[id_1]))
        else:
            print("MISMATCH of \"{}\" and \"{}\"!".format(labels[id_1],labels[id_2]))
    elif conf_1 <= 50:
        print("couldn't predict picture 2, confidence:{}".format(conf_2))
    elif conf_2 <= 50:
        print("couldn't predict picture 1, confidence:{}".format(conf_1))
    else:
        print("couldn't predict both photos")

    cv2.waitKey(0)
    cv2.destroyAllWindows()