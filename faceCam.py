import numpy as np
import cv2
import pickle


def faceCam():
    # Haar Cascade function: Face detection of the image
    # Create the haar cascade

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("trainner.yml")

    labels = {}

    with open("labels.pickle", "rb") as f:
        oldlabels = pickle.load(f)
        labels = {v:k for k,v in oldlabels.items()}

    # turn on webcam
    cap = cv2.VideoCapture(0)

    while(True):
        # capture by frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.5,
            minNeighbors = 5
        )

        # box around face
        for x,y,w,h in faces:

            roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w] 

            id , conf = recognizer.predict(roi_gray)
            
            if conf >= 45:
                print(id)
                print(labels[id])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


            # display the rectangle around the faces in the image
            color = (255,0,0) #BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

        # display the webcam
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()