#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:22:03 2021

@author: liyueyi
"""
import numpy as np
import cv2
import mediapipe as mp

class Detector():
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(False, 1, 0.85, 0.5)
    def getPosition(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
            return self.lmList
header = cv2.imread('1.png')
drawColor = (255, 255, 255) # default color
cap = cv2.VideoCapture(0)  # video device linux -1, win 0/1
cap.set(3, 1280)
cap.set(4, 720)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

detector = Detector()

brushWidth=10
x0, y0 = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lmList = detector.getPosition(img)

    if lmList!=None and len(lmList) != 0:
        x1, y1 = lmList[8][1:]  
        if x0 == 0 and y0 == 0:
            x0, y0 = x1, y1
        if y1 < 80:
            if 100 < x1 < 450:
                drawColor = (0, 0, 255)
            elif 500 < x1 < 850:
                drawColor = (0, 255, 0)
            elif 900 < x1 < 1250:
                drawColor = (255, 0, 0)
        cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
        cv2.line(img, (x0, y0), (x1, y1), drawColor, brushWidth)
        cv2.line(imgCanvas, (x0, y0), (x1, y1),
                     drawColor, brushWidth)
        x0, y0 = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)  
    img = cv2.bitwise_or(img, imgCanvas)
    
    img[0:100, 0:1280] = header
    cv2.imshow("FingerPainting", img)
    cv2.waitKey(1)
    