import cv2    # Add an image in the window

def faceDetect(imagePath):
    
    cascPath = "haarcascade_frontalface_default.xml"


    # Haar Cascade function: Face detection of the image
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)


    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Detect faces in the image
    # Return a rectangle with coordinates x,y,w,h around the dected face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )


    print("Found {0} faces!".format(len(faces)))


    # Display the rectangle around the faces in the image
    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)


    # Add image to the window
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)