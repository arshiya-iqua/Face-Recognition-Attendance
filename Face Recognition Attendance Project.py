from quopri import encode

import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

path='ImageAttendance'
images=[]
classNames=[]   
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('markAttendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
encodeListKnown=findEncodings(images)

cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)
    
    if len(encodesCurFrame)==0:
        print("No face detected in the current frame.")

    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            
        cv2.imshow('webcam',img)
        cv2.waitKey(1)
        y1,x2,y2,x1=faceLoc
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        if matches[matchIndex]:
            markAttendance(name)