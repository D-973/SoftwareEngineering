# Import Packages
import cv2
import numpy as np
import os 

# Set Current Working Directory
work_dir = "D:\Dhea\ML\FaceRecogDir";

# Take all facial samples on dataset directory, returning 2 arrays
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(work_dir+'/trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("{0}/haarcascade_frontalface_default.xml".format(work_dir))
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# Name related to ids: example ==> Semmy: id=1,  etc
names = ['None', 'Dhea', 'Atha', 'Hizkia'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Video rotation
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)),)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 50):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    # Show video
    cv2.imshow('Face Recognition using OpenCV',img) 
    
    # Handle key press
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup object.")
cam.release()
cv2.destroyAllWindows()