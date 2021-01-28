import cv2
import numpy as np
import face_recognition
import os

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for name in myList:
    currImg = cv2.imread(f'{path}/{name}')
    images.append(currImg)
    classNames.append(os.path.splitext(name)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodedList = findEncodings(images)
print('Encoding Complete')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.5,0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)
    for encodeFace,faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodedList,encodeFace)
        faceDis = face_recognition.face_distance(encodedList,encodeFace)
        matchIndex = np.argmin(faceDis)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*2,x2*2,y2*2,x1*2
        if faceDis[matchIndex]< 0.50:
            name = classNames[matchIndex].upper()
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,175),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

        cv2.rectangle(img,(x1,y1),(x2,y2),(100,100,100),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
