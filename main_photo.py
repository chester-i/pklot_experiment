from sre_constants import SUCCESS
import cv2
import pickle
from cv2 import ADAPTIVE_THRESH_GAUSSIAN_C
import cvzone
import numpy as np
import pandas as pd

# import train img file
file = "2015-11-16_1140.jpg"

# 처음 CarParkPos.txt import

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
        """성능을 올리기 위해서, 이 곳에서 더 많은 실험 예정"""
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

while loop_num == True:  # 영상이 아니기 때문에 딱 한 번만 돌 수 있게

    #success, img = cap.read()
    img = cv2.imread("opencv_train_cam1_imgs/"+file)
    # to check img size to normalize boundign box coordinates
    print(img.shape)
    size_x = img.shape[0]
    size_y = img.shape[1]

    # gray scale images to check if it's occupied
    # gray scale -> check edges and corners and
    # using threshold
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
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

    # export 라벨링 정보 in txt file
    cls_df['class'] = pos_cls[:len(posList)]  # 라벨링에 클래스 추가
    # yolov5 에 맞게 column 순서 설정
    cls_df = cls_df[['class', 0, 1, 2, 3, 4, 5, 6, 7]]

    # normalization to use yolov5
    cls_df.columns = ['cls', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    for col in cls_df.columns:
        if "x" in col:
            cls_df[col] = cls_df[col].apply(lambda x: x/size_x)
        elif "y" in col:
            cls_df[col] = cls_df[col].apply(lambda y: y/size_y)
    cls_df.to_csv("opencv_train_cam1_lbs/"+file[:-4]+'.txt', sep=" ", header=None,
                  index=None)  # 라벨링 txt file export
    #
    cv2.waitKey(10000000)  # 10000 mili seconds 뒤에 이미지 파일 종료

    loop_num = False
