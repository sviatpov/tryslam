import cv2
import numpy as np

def drawlines(img1, lines,pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r = img1.shape
    for r,pt1 in zip(lines,pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [20, -(r[2]+r[0]*20)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        # img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    return img1
