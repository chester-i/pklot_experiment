# selecting parking spots
import cv2
import pickle  # to save all of the places and the positions of parking psots and bring them to the main code
import numpy as np
import pandas as pd


# 처음 CarParkPos.txt import
try:
    df = pd.read_csv('CarParkPos.txt', sep=" ", header=None)
    posList = []
    for row in df.index.tolist():
        add = []
        for col in range(0, 7, 2):
            add.append([df.loc[row, col], df.loc[row, col+1]])
        posList.append(np.array(add))
    #print("add: ", add)
    print("initial posList: ", posList)

except:
    posList = []

posList2 = []
cnt = 0


def mouseClick(events, x, y, flags, params):
    global cnt
    if events == cv2.EVENT_LBUTTONDOWN:
        if x > size_y:
            print(img.shape)
            raise "x > size_x"
        if y > size_x:
            print(img.shape)
            raise "y > size_y"

        if cnt < 4:
            posList2.append((x, y))
            cnt += 1
        if cnt == 4:
            posList.append(np.array(posList2))
            cnt = 0
            print("cnt :", cnt, "postList :", posList)
            print("posList2 :", posList2)
            posList2.clear()

            #print("posList type: ", type(posList[0][0][0]))
            # add poplist to txt file
            # with open('CarParkPos.txt', 'w') as f:
            #    for i in posList:
            #        for j in range(0, 4):
            #            for row in [0, 1]:
            #                f.write(str(i[j][row]))
            #        f.write("\n")
            #    f.close()

    if events == cv2.EVENT_RBUTTONDOWN:
        print("right button clikced! coordinates are: ", x, y)
        xs_ys_i = {}
        for i, pos in enumerate(posList):

            # 오른쪽 클릭해서 찍은 곳(x,y)이 한 바운딩박스의 안에 있으면 해당 바운딩박스 삭제
            xs = []
            ys = []
            xs_ys_i[i] = {}
            for crnt in pos:
                xs.append(crnt[0])
                ys.append(crnt[1])

            max_x = max(xs)
            min_x = min(xs)
            max_y = max(ys)
            min_y = min(ys)
            xs_ys_i[i]['min_x'] = min_x
            xs_ys_i[i]['max_x'] = max_x
            xs_ys_i[i]['min_y'] = min_y
            xs_ys_i[i]['max_y'] = max_y

            ls = pos

            # print("min_x:", min_x, "max_x:", max_x,
            #      "min_y:", min_y, "max_y:", max_y)
            if (xs_ys_i[i]["min_x"] < x < xs_ys_i[i]['max_x']) and (xs_ys_i[i]['min_y'] < y < xs_ys_i[i]['max_y']):
                posList.pop(i)
                break  # 한번 클릭에 한 바운딩박스만 삭제
                print("ith position is eliminated")

    # add popList to txt file
    with open('CarParkPos.txt', 'w') as f:
        for pos in posList:
            add = ""
            for row in range(0, 4):
                for i in [0, 1]:
                    add += str(pos[row][i]) + " "  # separator는 빈칸
            f.writelines(add[:-1])  # 마지막 separator는 제거
            f.write("\n")
        f.close()


while True:
    img = cv2.imread('boxes1.jpg')
    global size_x, size_y
    size_x = img.shape[0]  # 클릭한 곳이 이미지의 x 최대값 벗어나지 않게
    size_y = img.shape[1]  # 클릭한 곳이 이미지의 y 최대값 벗어나지 않게

    # display rectangles
    for pos in posList:
        # instead of rectangle bounding boxes, we draw polylines of them
        cv2.polylines(img, np.int32([pos]), True, (255, 0, 255), 2)

    cv2.imshow("Image", img)  # open image
    # detect mouse click
    # mouseClick is a function where a mouse clicks
    cv2.setMouseCallback("Image", mouseClick)

    cv2.waitKey(1)
