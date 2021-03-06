from sre_constants import SUCCESS
import cv2
import pickle
from cv2 import ADAPTIVE_THRESH_GAUSSIAN_C
import cvzone
import numpy as np
import pandas as pd
import os
img_lists = os.listdir("opencv_train_cam1_imgs/")
# import train img file


def imgs_to_lbs(file):

    df = pd.read_csv('CarParkPos.txt', sep=" ", header=None)
    posList = []
    for row in df.index.tolist():
        add = []
        for col in range(0, 7, 2):
            add.append([df.loc[row, col], df.loc[row, col+1]])
        posList.append(np.array(add))

    # assign gloabl variable 'pos_col'
    pos_cls = []
    print("the 1st length of pos_cls: ", len(pos_cls))

    """with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)"""

    # crop all the regions

    # imgPro will be image processed by gray-scale, blurring, median blurring
    def checkParkingSpace(imgPro):
        spaceCounter = 0
        print("checking parking space process starts")
        for pos in posList:
            # crop by polygon
            rect = cv2.boundingRect(pos)
            x, y, w, h = rect
            cropped = imgPro[y:y+h, x:x+w].copy()

            pts = pos - pos.min(axis=0)

            mask = np.zeros(cropped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            dst = cv2.bitwise_and(cropped, cropped, mask=mask)

            imgCrop = dst

            # to check its occupancy
            # adding more expirements
            count = cv2.countNonZero(imgCrop)
            count2 = np.sum(dst == 0)
            portion = count2/(count+count2)

            #
            cvzone.putTextRect(img, str(count), (x, y), scale=1,
                               thickness=2, offset=0, colorR=(0, 0, 255))

            if count < 60:  # it's empty
                color = (0, 255, 0)
                thickness = 2
                spaceCounter += 1
                pos_cls.append(4)
            else:
                color = (0, 0, 255)
                thickness = 2
                pos_cls.append(3)

            cv2.polylines(img, np.int32([pos]), True, color, thickness)

        # show the number of empty and occupied spots
        cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50),
                           scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

    loop_num = True

    cls_df = pd.read_csv('CarParkPos.txt', sep=" ", header=None)

    while loop_num == True:  # ????????? ????????? ????????? ??? ??? ?????? ??? ??? ??????

        #success, img = cap.read()
        img = cv2.imread("opencv_train_cam1_imgs/"+file)
        # to check img size to normalize boundign box coordinates
        print(img.shape)
        size_x = img.shape[1]
        size_y = img.shape[0]

        # gray scale images to check if it's occupied
        # gray scale -> check edges and corners and
        # using threshold
        # convert to gray scale
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)  # (3,3), 1 can change
        # after blurring, we go to binary image
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        # to remove salt and pepper noise, use median blur
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        # makes it thicker
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        checkParkingSpace(imgDilate)

        cv2.imshow("Image", img)
        cv2.imshow("ImageBlur", imgBlur)
        cv2.imshow("ImageThres", imgDilate)

        # export ????????? ?????? in txt file
        cls_df['class'] = pos_cls[:len(posList)]  # ???????????? ????????? ??????
        # yolov5 ??? ?????? column ?????? ??????
        cls_df = cls_df[['class', 0, 1, 2, 3, 4, 5, 6, 7]]

        # normalization to use yolov5
        cls_df.columns = ['cls', 'x1', 'y1',
                          'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        for col in cls_df.columns:
            if "x" in col:
                cls_df[col] = cls_df[col].apply(lambda x: x/size_x)
            elif "y" in col:
                cls_df[col] = cls_df[col].apply(lambda y: y/size_y)
        cls_df.to_csv("opencv_train_cam1_lbs/"+file[:-4]+'.txt', sep=" ", header=None,
                      index=None)  # ????????? txt file export
        #
        cv2.waitKey(1)  # 10000 mili seconds ?????? ????????? ?????? ??????

        loop_num = False


for img_name in img_lists:
    imgs_to_lbs(img_name)
