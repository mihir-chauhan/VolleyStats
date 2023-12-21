import PIL
from tkinter import *
import customtkinter
from PIL import Image, ImageTk, ImageFilter
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO
import cv2
import skimage
import mahotas
import mahotas.demos
import math

firstFrame = True

panelA = None
imageViewSize = (1920/2, 1080/2)

def createUI():
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    global root
    root = customtkinter.CTk()
    root.geometry("500x350")

    frame = customtkinter.CTkFrame(master = root)
    frame.pack (pady = 20, padx = 60, fill = "both", expand = False)

    button2 = customtkinter.CTkButton(master = frame, text = "-", command = nextBBox)
    
    button2.pack (pady = 12, padx = 10)
        
    image = cv2.imread("temp.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)
    
    
    button_image = ImageTk.PhotoImage(image1, size=imageViewSize)

    global imageView

    imageView = Label(image=button_image)
    imageView.image = button_image
    imageView.pack(padx=20, pady=10)

    root.mainloop()


def nextBBox():
    global firstFrame
    global root
    firstFrame = not firstFrame
    print(firstFrame)

    image = cv2.imread("prediction.jpg" if firstFrame else "temp.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image)

    button_image = ImageTk.PhotoImage(image1, size=(500, 500))


    global imageView
    imageView.configure(image=button_image)
    imageView.image = button_image


    # global button
    # button.pack_forget()
    # button = customtkinter.CTkLabel(root, image=button_image, text="")
    root.mainloop()


createUI()