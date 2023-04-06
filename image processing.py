# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:05:22 2022

@author: Berfin
"""

import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from abc import ABC, abstractmethod

root = Tk()
root.geometry('1000x1000')
root.configure(bg='black')
root.title('THKU Term Project')
Label(root,text='Berfin Sine 190444051',font=50,bg='black',fg='red').pack()

f1 = LabelFrame(root,bg= 'blue')
f1.place(x=50,y=100)
L1 = Label(f1)
L1.pack()
f2 = LabelFrame(root,bg= 'blue')
f2.place(x=226,y=100)
L2 = Label(f2)
L2.pack()
f3 = LabelFrame(root,bg= 'blue')
f3.place(x=402,y=100)
L3 = Label(f3)
L3.pack()

def calculate():
    print("Pixel No: ")


def apply():
    print('Filter:')

b1 = Button(root, text='Apply Threshold', width=20,height= 2, bg='red', fg='white', command=apply)
b1.place(x=650, y=150)

b2 = Button(root, text='Calculate for Region 1', width=20,height= 2, bg='light blue', fg='black', command=calculate)
b2.place(x=650, y=275)

b3 = Button(root, text='Calculate for Region 2', width=20,height= 2, bg='light blue', fg='black', command=calculate)
b3.place(x=650, y=400)
b4 = Button(root, text='Calculate for Region 3', width=20,height= 2, bg='light blue', fg='black', command=calculate)
b4.place(x=650, y=525)


class Frame_grab(ABC):
    @abstractmethod
    def __init__(self,cap):
        self.cap=cap
    cap = cv2.VideoCapture(0)


    cv2.namedWindow('Trackbars')

    def callback_val(x):
        print(x)

    cv2.createTrackbar('LH', 'Trackbars', 0, 255, callback_val)
    cv2.createTrackbar('UH', 'Trackbars', 255, 255, callback_val)
    cv2.createTrackbar('LS', 'Trackbars', 0, 255, callback_val)
    cv2.createTrackbar('US', 'Trackbars', 255, 255, callback_val)
    cv2.createTrackbar('LV', 'Trackbars', 0, 255, callback_val)
    cv2.createTrackbar('UV', 'Trackbars', 255, 255, callback_val)
    cv2.createTrackbar('Threshold', 'Trackbars', 0, 255, callback_val)



    while True:

        region1 = cap.read()[1]
        region2 = cap.read()[1]
        region3 = cap.read()[1]



        region1 = region1[0:528, 0:176]
        region2 = region2[0:528, 177:352]
        region3 = region3[0:528, 353:528]

        region1_1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
        region2_1 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)
        region3_1 = cv2.cvtColor(region3, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos('LH', 'Trackbars')
        l_s = cv2.getTrackbarPos('LS', 'Trackbars')
        l_v = cv2.getTrackbarPos('LV', 'Trackbars')

        u_h = cv2.getTrackbarPos('UH', 'Trackbars')
        u_s = cv2.getTrackbarPos('US', 'Trackbars')
        u_v = cv2.getTrackbarPos('UV', 'Trackbars')

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask1 = cv2.inRange(region1_1, l_b, u_b)
        res1 = cv2.bitwise_and(region1_1, region1_1, mask=mask1)

        mask2 = cv2.inRange(region2_1, l_b, u_b)
        res2 = cv2.bitwise_and(region2_1, region2_1, mask=mask2)

        mask3 = cv2.inRange(region3_1, l_b, u_b)
        res3 = cv2.bitwise_and(region3_1, region3_1, mask=mask3)

        region1 = ImageTk.PhotoImage(Image.fromarray(res1))
        region2 = ImageTk.PhotoImage(Image.fromarray(res2))
        region3 = ImageTk.PhotoImage(Image.fromarray(res3))

        L1["image"] = region1
        L2["image"] = region2
        L3["image"] = region3

        

        root.update()

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

class Sliders_Data(ABC):
    __trackbar = 'trackbar'

    __trackBars = []

    cv2.namedWindow(__trackbar)

    def Set_Slider_Data(self):
        __trackBars = [
            cv2.createTrackbar('LH', self.__trackbar, 0, 255, onChange),
            cv2.createTrackbar('UH', self.__trackbar, 255, 255, onChange),
            cv2.createTrackbar('LS', self.__trackbar, 0, 255, onChange),
            cv2.createTrackbar('US', self.__trackbar, 255, 255, onChange),
            cv2.createTrackbar('LV', self.__trackbar, 0, 255, onChange),
            cv2.createTrackbar('UV', self.__trackbar, 255, 255, onChange),
            cv2.createTrackbar('Threshold', self.__trackbar, 0, 255, onChange)]

    def Get_Slider_Data(self):
        lh = cv2.getTrackbarPos('LH', self.__trackbar)
        uh = cv2.getTrackbarPos('UH', self.__trackbar)
        ls = cv2.getTrackbarPos('LS', self.__trackbar)
        us = cv2.getTrackbarPos('US', self.__trackbar)
        lv = cv2.getTrackbarPos('LV', self.__trackbar)
        uv = cv2.getTrackbarPos('UV', self.__trackbar)
        th_val = cv2.getTrackbarPos('Threshold', self.__trackbar)

        lower = np.array([lh, ls, lv], np.uint8)
        upper = np.array([uh, us, uv], np.uint8)

        return lower, upper, th_val

class Region1(Frame_Grab,Sliders_Data):
	def __init__(self):
		Region.__init__(self)

	def Pixel_Calculation(self, img):
		slider_data = self.Get_Slider_Data()
		lowerHSV = np.array([slider_data[0], slider_data[2], slider_data[4]])
		upperHSV = np.array([slider_data[1], slider_data[3], slider_data[5]])

		img = self.ConvertIntoHSV(img, np.array(lowerHSV), np.array(upperHSV))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.PixelNo = cv2.countNonZero(img)

class Region2(Frame_Grab,Sliders_Data):
	def __init__(self):
		Region.__init__(self)

	def Pixel_Calculation(self, img):
		slider_data = self.Get_Slider_Data()
		lowerHSV = np.array([slider_data[0], slider_data[2], slider_data[4]])
		upperHSV = np.array([slider_data[1], slider_data[3], slider_data[5]])

		img = self.ConvertIntoHSV(img, np.array(lowerHSV), np.array(upperHSV))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.PixelNo = cv2.countNonZero(img)

class Region3(Frame_Grab,Sliders_Data):
	def __init__(self):
		Region.__init__(self)

	def Pixel_Calculation(self, img):
		slider_data = self.Get_Slider_Data()
		lowerHSV = np.array([slider_data[0], slider_data[2], slider_data[4]])
		upperHSV = np.array([slider_data[1], slider_data[3], slider_data[5]])

		img = self.ConvertIntoHSV(img, np.array(lowerHSV), np.array(upperHSV))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.PixelNo = cv2.countNonZero(img)

class Filter(Frame_Grab, Sliders_Data):
	def __init__(self):
		Frame_Grab.__init__(self)
		Sliders_Data.__init__(self)

		self.Set_Slider_Data(255)

	def Filter_Implement(self, img):
		_, res = cv2.threshold(img, self.Get_Slider_Data(), 255, cv2.THRESH_TRUNC)
		return res