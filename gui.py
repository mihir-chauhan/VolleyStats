from tkinter import *
import customtkinter
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO
import cv2
import math

imageViewSize = (1920/2, 1080/2)

def submit():
    global jerseyEntry
    stored = jerseyEntry.get()
    global jerseyNumbersList
    global indexOfBBox
    jerseyNumbersList[indexOfBBox] = stored
    global previousBBoxes
    global currentBBoxes
    global predictedBBoxes
    previousBBoxes[stored] = previousBBoxes[indexOfBBox + 100]
    del previousBBoxes[indexOfBBox + 100]
    currentBBoxes[stored] = currentBBoxes[indexOfBBox + 100]
    del currentBBoxes[indexOfBBox + 100]
    predictedBBoxes[stored] = predictedBBoxes[indexOfBBox + 100]
    del predictedBBoxes[indexOfBBox + 100]
    jerseyEntry.delete(0, END)
    newJerseyNumbersList = {}
    for jNum in jerseyNumbersList:
        if(jerseyNumbersList[jNum] != ''):
            newJerseyNumbersList[jNum] = jerseyNumbersList[jNum]
    jerseyNumbersList = newJerseyNumbersList
    drawBBox(indexOfBBox)
    
def createUI():
    global previousBBoxes
    previousBBoxes = {}
    global currentBBoxes
    currentBBoxes = {}
    global predictedBBoxes
    predictedBBoxes = {}
    global predictedBBoxesHolder
    predictedBBoxesHolder = {}
    global jerseyNumbersList
    jerseyNumbersList = {}
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue") 
    global root
    root = customtkinter.CTk()
    root.title("Noodles AI")
    global frame
    frame = customtkinter.CTkFrame(master = root)
    frame.pack (pady = 20, padx = 60, fill = "both", expand = False)
    global prevButton
    global nextButton
    prevButton = customtkinter.CTkButton(master = frame, text = "<", command = previousBBox)
    nextButton = customtkinter.CTkButton(master = frame, text = ">", command = nextBBox)
    prevButton.grid(column=0, row=0, pady = 12, padx = 12)
    nextButton.grid(column=1, row=0, pady = 12, padx = 12)
    splash = cv2.imread("splash.jpg")
    cv2.imwrite('temp.jpg', splash) 
    global image
    image = cv2.imread("temp.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    global imageView
    view_image = ImageTk.PhotoImage(image1, size=imageViewSize)
    imageView = Label(image=view_image)
    imageView.image = view_image
    imageView.pack(padx=0, pady=0)
    global jerseyEntry
    jerseyEntry = customtkinter.CTkEntry(master = frame, placeholder_text = "Enter in number", width=200)
    jerseyEntry.grid(column=3, row = 0, pady = 12, padx = 12)
    global submitBtn
    submitBtn = customtkinter.CTkButton(master = frame, text = "Submit", command=submit)
    submitBtn.grid(column = 4, row = 0, pady = 12, padx = 12)
    global uploadBtn
    uploadBtn = customtkinter.CTkButton(master=frame, text="Upload", command=upload)
    uploadBtn.grid (column = 5, row = 0, pady = 12, padx = 24)
    global canRun
    canRun = True
    btn = customtkinter.CTkButton(master=frame, text="Run", command=runOnLoop)
    btn.grid (column = 6, row = 0, pady = 12, padx = 24)
    root.mainloop()
    
def runOnLoop():
    global canRun
    global root
    global frame
    canRun = True
    global jerseyEntry
    global submitBtn
    global uploadBtn
    global prevButton
    global nextButton
    uploadBtn.grid_forget()
    prevButton.grid_forget()
    nextButton.grid_forget()
    submitBtn.grid_forget()
    jerseyEntry.grid_forget()
    btn2 = customtkinter.CTkButton(master=frame, text="Pause", command=stopRun)
    btn2.grid (column = 7, row = 0, pady = 12, padx = 24)
    root.update()
    runYOLONotFirstRun()


def stopRun():
    global canRun
    canRun = False

def runYOLONotFirstRun():
    global root
    runYOLO(drawBBoxes=False)
    global canRun
    if canRun:
        root.after(0, runYOLONotFirstRun)

def bboxDistance(bbox1, bbox2):
    return (math.sqrt((math.pow(bbox1[0] - bbox2[0], 2)) + (math.pow(bbox1[1] - bbox2[1], 2))))

def upload():
    global vidcap
    global playerModel
    global actionModel
    global panelA, panelB, image
    f_types = [('mp4 files', '*.mp4')]
    path = filedialog.askopenfilename(filetypes=f_types)
    playerModel = YOLO('player_det.pt')
    actionModel = YOLO('action_det.pt')
    vidcap = cv2.VideoCapture(path)
    runYOLO(drawBBoxes=True)

def nextBBox():
    global resultxyxys
    global indexOfBBox
    if indexOfBBox < len(resultxyxys) - 1:
        indexOfBBox += 1
    drawBBox(indexOfBBox=indexOfBBox)

def previousBBox():
    global indexOfBBox
    if indexOfBBox > 0:
        indexOfBBox -= 1
    drawBBox(indexOfBBox=indexOfBBox)


def drawBBox(indexOfBBox):
    global resultxyxys
    global image
    global root
    global jerseyNumbersList
    imageLocal = cv2.rectangle(image.copy(), (0,0), (0,0),color=(255,255,255), thickness=1)
    for i in range(len(resultxyxys)):
        color = (255,255,255)
        thickness = 5
        if(i == indexOfBBox):
            color = (255,255,255)
            thickness = 5
        else:
            color = (95,95,95)
            thickness = 1
        cv2.rectangle(imageLocal, (int(resultxyxys[i][0][0]), int(resultxyxys[i][0][1])), (int(resultxyxys[i][0][2]), int(resultxyxys[i][0][3])),color=color, thickness=thickness)
        if i in jerseyNumbersList:
            cv2.putText(imageLocal, jerseyNumbersList.get(i), (int(resultxyxys[i][0][0]), int(resultxyxys[i][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    imageLocal = cv2.cvtColor(imageLocal, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(imageLocal)
    global imageView
    view_image = ImageTk.PhotoImage(image1, size=imageViewSize)
    imageView.configure(image=view_image)
    imageView.image = ImageTk.PhotoImage(image1, size=imageViewSize)
    root.mainloop()

def updateImageNoBBox():
    global resultxyxys
    global image
    global root
    imageLocal = cv2.rectangle(image.copy(), (0,0), (0,0),color=(255,255,255), thickness=1)
    imageLocal = cv2.cvtColor(imageLocal, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(imageLocal)
    global imageView
    view_image = ImageTk.PhotoImage(image1, size=imageViewSize)
    imageView.configure(image=view_image)
    imageView.image = ImageTk.PhotoImage(image1, size=imageViewSize)
    root.update()

def runYOLO(drawBBoxes):
    global indexOfBBox
    global vidcap
    indexOfBBox = 0
    hasFrame,frame = vidcap.read()
    global resultxyxys
    global image
    global root
    global previousBBoxes
    global currentBBoxes
    global predictedBBoxes
    global predictedBBoxesHolder
    global jerseyNumbersList 

    resultxyxys = []
    actionResultsxyxys = []
    if hasFrame:
            cv2.imwrite("temp.jpg", frame)
            results = playerModel.predict('temp.jpg', save=True, imgsz=1080, conf=0.5, project="./test/", exist_ok=True, show=False, verbose=False)
            actionResults = actionModel.predict('temp.jpg', save=True, imgsz=1080, conf=0.5, project="./test/", exist_ok=True, show=False, verbose=False)
            names = playerModel.names
            #{0: 'block', 1: 'defense', 2: 'serve', 3: 'set', 4: 'spike'}
            image = cv2.imread('temp.jpg')
            cv2.resize(image, (640, 1088))
            for r in results:
                for c in r.boxes:
                    if (names[int(c.cls)]) == "person":
                        resultxyxys.append(c.xyxy)
            for r in actionResults:
                for c in r.boxes:
                    actionResultsxyxys.append(c.xyxy)
            previousBBoxes = currentBBoxes
            currentBBoxes = {}
            inti = 0
            for r in resultxyxys:
                inti += 1
                jerseyNumber = -1
                if drawBBoxes:
                    jerseyNumber = inti - 1 + 100
                    previousBBoxes[jerseyNumber] = ([int(r[0][0]), int(r[0][1]), int(r[0][2]), int(r[0][3]), 
                                        (int(r[0][0]) + int(r[0][2]))/2.0, (int(r[0][1]) + int(r[0][3]))/2.0])
                    currentBBoxes[jerseyNumber] = ([int(r[0][0]), int(r[0][1]), int(r[0][2]), int(r[0][3]), 
                                        (int(r[0][0]) + int(r[0][2]))/2.0, (int(r[0][1]) + int(r[0][3]))/2.0])
                else:
                    distances = []
                    jerseyNumbers = []
                    for predictedBox in predictedBBoxes:
                        distances.append(bboxDistance([(int(r[0][0]) + int(r[0][2]))/2.0, (int(r[0][1]) + int(r[0][3]))/2.0], predictedBBoxes[predictedBox]))
                        jerseyNumbers.append(predictedBox)
                    jerseyNumber = jerseyNumbers[distances.index(np.min(distances))]
                    currentBBoxes[jerseyNumber] = ([int(r[0][0]), int(r[0][1]), int(r[0][2]), int(r[0][3]), 
                                        (int(r[0][0]) + int(r[0][2]))/2.0, (int(r[0][1]) + int(r[0][3]))/2.0])
                predictedBBoxesHolder[jerseyNumber] = ([(2 * currentBBoxes[jerseyNumber][4]) - previousBBoxes[jerseyNumber][4], (2 * currentBBoxes[jerseyNumber][5]) - previousBBoxes[jerseyNumber][5]])
            predictedBBoxes = predictedBBoxesHolder
            predictedBBoxesHolder = {}
            if drawBBoxes:
                drawBBox(indexOfBBox)
            else:
                updateImageNoBBox()

def destroy():
    vidcap.release()

createUI()