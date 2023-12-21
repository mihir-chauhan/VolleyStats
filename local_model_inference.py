import yolov5
import torch
from ultralytics import YOLO
import cv2
import math 
import os
import numpy as np
from roboflow import Roboflow



# TODO
# 1. First frame GUI - match label bbox to pkayer number
# 2. Action det (bringing in the model) - ez
# 3. Alert for - is player <> doing <>? Yes/No, Adjustment
# 4. Saving stats

model = YOLO('player_det.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images

print("File exists?", os.path.exists('Moreau Set 5.mp4'))

vidcap = cv2.VideoCapture('Moreau Set 5.mp4')

previousBBoxes = []
currentBBoxes = []
predictedBBoxes = []
predictedBBoxesHolder = []

firstFrame = True

def bboxDistance(bbox1, bbox2):
    return (math.sqrt((math.pow(bbox1[0] - bbox2[0], 2)) + (math.pow(bbox1[1] - bbox2[1], 2))))


# playerIndex 0
# = []
while True:
   hasFrame,frame = vidcap.read()
   if hasFrame:
        cv2.imwrite("temp.jpg", frame)
        # cv2.imshow('window_name', frame)
        results = model(['temp.jpg'])  # return a list of Results objects
        predictions = model.predict('temp.jpg', save=False, imgsz=1080, conf=0.5, project="./test/")
        names = model.names


        previousBBoxes = currentBBoxes
        currentBBoxes = []
        counter = 0
        for r in results:
            for c in r.boxes:
                if (names[int(c.cls)]) == "person":
                    if firstFrame:
                        # format: x1, y1, x2, y2, c_x, c_y
                        previousBBoxes.append([int(c.xyxy[0][0]), int(c.xyxy[0][1]), int(c.xyxy[0][2]), int(c.xyxy[0][3]), 
                                            (int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0, (int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0])
                        currentBBoxes.append([int(c.xyxy[0][0]), int(c.xyxy[0][1]), int(c.xyxy[0][2]), int(c.xyxy[0][3]), 
                                            (int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0, (int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0])
                    else:
                        currentBBoxes.append([int(c.xyxy[0][0]), int(c.xyxy[0][1]), int(c.xyxy[0][2]), int(c.xyxy[0][3]), 
                                            (int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0, (int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0])
                        
                        pt1 = (int((int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0 - 25), int((int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0 - 25))
                        pt2 = (int((int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0 + 25), int((int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0 + 25))
                        cv2.rectangle(frame, pt1, pt2, (36,255,12), 1)
                        cv2.imshow("window", frame)
                        distances = []
                        for predictedBox in predictedBBoxes:
                            # if bboxDistance([(int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0, (int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0], predictedBox) < 50:
                            #     #track
                            distances.append(bboxDistance([(int(c.xyxy[0][0]) + int(c.xyxy[0][2]))/2.0, (int(c.xyxy[0][1]) + int(c.xyxy[0][3]))/2.0], predictedBox))
                            cv2.rectangle(frame, pt1, pt2, (24,24,24), 1)
                        print("minDist: ", np.min(distances))
                        print("prediction: ", np.max(distances))
                        # playerIndex[r]

                    # print(currentBBoxes)
                    # print(previousBBoxes)
                    # FAILS WHEN YOU LOSE A BOUNDING BOX
                    if not(len(currentBBoxes) - 1 <= counter or len(previousBBoxes) - 1 <= counter):
                        predictedBBoxesHolder.append([(2 * currentBBoxes[counter][4]) - previousBBoxes[counter][4], (2 * currentBBoxes[counter][5]) - previousBBoxes[counter][5]])
                        
                        print(predictedBBoxesHolder)
                        print("a")
                        print(frame, ((2 * currentBBoxes[counter][4]) - previousBBoxes[counter][4], (2 * currentBBoxes[counter][5]) - previousBBoxes[counter][5]), ((2 * currentBBoxes[counter][4]) - previousBBoxes[counter][4] + 50, (2 * currentBBoxes[counter][5]) - previousBBoxes[counter][5] + 50), (36,255,12), 1)
                        counter+=1
        # cv2.waitKey()

        firstFrame = False
        predictedBBoxes = predictedBBoxesHolder
        predictedBBoxesHolder = []
   else:
      break

vidcap.release()






# ckpt = torch.load("yolov8m.pt", force_reload=True)  # applies to both official and custom models
# torch.save(ckpt, "updated-model.pt")

# model = yolov5.load('updated-model.pt')
  
# # set model parameters
# model.conf = 0.5  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 50  # maximum number of detections per image

# # set image
# img = 'temp.jpg'

# # perform inference
# results = model(img)

# # inference with larger input size
# results = model(img, size=1280)

# # inference with test time augmentation
# results = model(img, augment=True)

# # parse results
# predictions = results.pred[0]
# boxes = predictions[:, :4] # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]

# # show detection bounding boxes on image
# results.show()

# # save results into "results/" folder
# results.save(save_dir='results/')