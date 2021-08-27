import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
# from PIL import ImageGrab
tracks = {}

def markAttendance(name):
	if not tracks:
		tracks[name] = []
	else:
		if name not in tracks.keys():
			tracks[name] = []

	now = datetime.now()
	date = now.date()
	dtString = now.time()
	tracks[name].append(datetime.combine(date, dtString))

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
encodeListKnown = data["encodings"]
 
print("[INFO] Starting Camera...")
#cap = cv2.VideoCapture("http://10.190.234.177:8080/video")
cap = cv2.VideoCapture(0)
mul = 4
while True:
	success, img = cap.read()
	#img = captureScreen()
	imgS = cv2.resize(img,(0,0),None,0.25, 0.25)
	imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
	for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
		matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
		faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
		matchIndex = np.argmin(faceDis)
		if matches[matchIndex]:
			name = data["names"][matchIndex].upper()
		else:
			name = "unknown".upper()
		y1,x2,y2,x1 = faceLoc
		y1, x2, y2, x1 = round(y1*mul),round(x2*mul),round(y2*mul),round(x1*mul)
		cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
		cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
		cv2.putText(img,name,(x1+6, y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
		markAttendance(name)
	cv2.imshow('Webcam',img)
	key = cv2.waitKey(1) & 0xff
	if key == 27:
		break
cap.release()
for key, value in tracks.items():
	f = open("Attendance.csv",'a')
	name = key
	start = value[0]
	end = value[-1]
	date = str(start)[:10]
	time = end - start
	f.write("\n"+name+","+date+","+str(start)[12:]+","+str(end)[12:]+","+str(time))
	f.close()
print("[INFO] Goodbye...")
