import math

import mediapipe as mp
import cv2
import time

import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.SetMasterVolumeLevel(-65.25, None)
print(volume.GetMute())
print(volume.GetMasterVolumeLevel())
print(volume.GetVolumeRange())

cam = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
hand = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils


def findSlope(x1, x2, y1, y2):
    y = y1 - y2
    x = x1 - x2
    return math.atan(y / x)


while True:
    suc, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = np.fliplr(imgRGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    results = hand.process(imgRGB)
    f1 = ()
    f2 = ()
    f0 = ()
    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            draw.draw_landmarks(img, handlm, mpHand.HAND_CONNECTIONS)
            for id, lm in enumerate(handlm.landmark):
                if id == 8:
                    h, w, c = img.shape
                    c = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, c, 8, (255, 0, 255), cv2.FILLED)
                    f2 = c
                if id == 4:
                    h, w, c = img.shape
                    c = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, c, 8, (255, 0, 255), cv2.FILLED)
                    f1 = c
                if id == 0:
                    h, w, c = img.shape
                    c = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, c, 8, (255, 0, 255), cv2.FILLED)
                    f0 = c

            cv2.line(img, f1, f2, (0, 255, 0), 2)

            x1, y1 = f1
            x2, y2 = f2
            x0, y0 = f0
            cv2.line(img, f1, f0, (255, 0, 0), 1)
            cv2.line(img, f2, f0, (255, 0, 0), 1)
            slope = (y0 - y2)

            print(np.abs(findSlope(x1, x0, y1, y0) - findSlope(x2, x0, y2, y0)))
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 8, (255, 50, 50), cv2.FILLED)
            length = math.hypot(x1 - x2, y1 - y2)
            vol = np.interp(length, [15, 120], [-65.25, 0])
            # print(length,vol)
            # volume.SetMasterVolumeLevel(vol,None)
            bar = np.interp(length, [15, 120], [300, 15])
            # print(bar)
            cv2.rectangle(img, (15, int(bar)), (40, 300), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (15, 15), (40, 300), (0, 255, 0), 3)
    cv2.imshow("image", img)

    cv2.waitKey(1)
