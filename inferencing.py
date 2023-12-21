from roboflow import Roboflow 
import cv2
import os
import json

print("File exists?", os.path.exists('Moreau Set 5.mp4'))

rf = Roboflow(api_key="mUWNY4Fdbhd3W14OCyoM")
actionsProject = rf.workspace().project("volleyball-actions")
actionDetModel = actionsProject.version(4).model 

playersProject = rf.workspace().project("players-dataset")
playersDetModel = playersProject.version(2).model

vidcap = cv2.VideoCapture('Moreau Set 5.mp4')

for i in range(35):
   hasFrame,frame = vidcap.read()


while True:
   hasFrame,frame = vidcap.read()
   if hasFrame:
        cv2.imwrite("temp.jpg", frame)
        cv2.imshow('window_name', frame)
        actionsJSON = actionDetModel.predict("temp.jpg", confidence=40, overlap=30).json()
        playersJSON = playersDetModel.predict("temp.jpg", confidence=40, overlap=30).json()
        annotatedImage = frame
        for i in actionsJSON["predictions"]:
            print(+i["width"])
            annotated = cv2.rectangle(annotatedImage, (i["x"], i["y"]), (i["x"] + i["width"], i["y"] + i["height"]), (36,255,12), 1)

        playersDetModel.predict("temp.jpg", confidence=40, overlap=30).save("prediction.jpg") # saves inferenced annoted file
   else:
      break

vidcap.release()