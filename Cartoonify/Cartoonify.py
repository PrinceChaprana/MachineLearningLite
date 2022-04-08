#Open cV import for computer vision
from email import message
import cv2
#easygui to open a file box to select a file from system
import easygui
#to work upon the file selected by easygui
import imageio
#
import numpy as np
#
import sys
#
import matplotlib.pyplot as plt
#
import os
#
import tkinter as tk
#
from tkinter import filedialog
#
from tkinter import *
#pillow to python image Library
from PIL import ImageTk,Image

#file box to choose the file
ImagePath = ""
Resized6 = ""
def upload():
    #this code will use the file box everytime we run it
    global ImagePath
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)

def cartoonify(ImagePath):
    #read the image
    #this will store the image in form of numbers help to perform operation
    originalImage = cv2.imread(ImagePath)
    originalImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2RGB)

    if originalImage is None:
        print("NO file exit")
        sys.exit()
    #resize the image with each transformation
    Resized1 = cv2.resize(originalImage,(960,540))

    #step1 convert the image to grayscale
    grayscaleImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2GRAY)
    #again resize the image to convert to same size
    Resized2 = cv2.resize(grayscaleImage ,(960,540))

    #step2 Smooth the grayscale Image by Bluring it by assigning a mean value to all pixe which fall in that category
    smoothgrayscale = cv2.medianBlur(grayscaleImage,5)
    Resized3 = cv2.resize(smoothgrayscale,(940,540))

    #step4 retriving the edges of the images
    #9,9 is block size
    #adaptive thresh calculate mean of neighbour pixel values area and minus a constant c from it
    getEdge = cv2.adaptiveThreshold(smoothgrayscale,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    Resized4 = cv2.resize(getEdge ,(960,540))

    #in carton effect ethir we highlight edge or smooth color
    
    #Masking the image to smooth the color just like beautify or ai effect in camera
    colorImage = cv2.bilateralFilter(originalImage,9,300,300)
    Resized5 = cv2.resize(colorImage ,(960,540))

    #to give cartonny effect
    #just like merge of two layer in photshop 
    cartoonImage = cv2.bitwise_and(colorImage,colorImage,mask = getEdge)
    global Resized6
    Resized6 = cv2.resize(cartoonImage ,(960,540))

    #to plot all transistion together
    images = [Resized1,Resized2,Resized3,Resized4,Resized5,Resized6]
    fig,axes = plt.subplots(3,2,figsize = (8,8),subplot_kw = {'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i],cmap = 'gray')
    plt.show()

def save(ImagePath,Resized6):
    newName = "catoonImage"
    path1 = os.path.dirname(ImagePath)
    extension = os.path.splitext(ImagePath)[1]
    path = os.path.join(path1,newName+extension)
    cv2.imwrite(path,cv2.cvtColor(Resized6,cv2.COLOR_RGB2BGR))
    I = "Image is saved at"+path
    tk.messagebox.showinfo(title=None,message=I)

top=tk.Tk()
top.geometry('400x400')
top.title('Cartoonify Your Image !')
top.configure(background='white')
label=Label(top,background='#CDCDCD', font=('calibri',20,'bold'))

upload=Button(top,text="Cartoonify an Image",command=upload,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
upload.pack(side=TOP,pady=50)

save1=Button(top,text="Save cartoon image",command=lambda: save(ImagePath, Resized6),padx=30,pady=5)
save1.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
save1.pack(side=TOP,pady=50)


top.mainloop()