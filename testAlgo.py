from ast import Not
# from asyncio.windows_events import NULL
from time import time

import cv2
import numpy as np
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import random

def detectAndDescribe(image, method='SIFT'):
    """
    Compute key points and feature descriptors using an specific method
    """
    # assert method is not None, "You need to define a feature detection method. Value
    # detect and extract features from the image
    if method == 'SIFT':
        # python 3.6
        descriptor = cv2.SIFT_create()
        # descriptor = cv2.SIFT_create()
        
        # kps = np.float32([kp.pt for kp in kps])
        # descriptor = cv2.SIFT_create()
    # elif method == 'SURF':
    #     descriptor = cv2.SURF_create()
        # kps = np.float32([kp.pt for kp in kps])
        # cv2.SURF_create()
        # descriptor = cv2.SURF_create()
    elif method == 'BRISK':
        descriptor = cv2.BRISK_create()
    elif method == 'ORB':
        descriptor = cv2.ORB_create()
    # get keypoints and descriptors
    
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"

    if method == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'ORB' or method == 'BRISK':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def draw_points(pic,points,name):
    for i in range(len(points[0])):
        cv2.circle(pic, (int(points[0][i][0]),int(points[0][i][1])), 2, (0, 255, 255), -1)
    cv2.imwrite(str(name)+".jpg",pic)

def draw_lines(pic1,pic2,ret,ret1,name):
    match_img = np.hstack([pic1, pic2])
    kp1 = ret['keypoints0'][0].cpu().numpy().astype(np.int)
    kp2 = ret1['keypoints1'][0].cpu().numpy().astype(np.int) 
    match1 = ret1['matches0'][0].cpu().numpy().astype(np.int)
    for i, p in enumerate(match1):
        if p!=-1:
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(match_img, (kp1[i][0], kp1[i][1]), (kp2[p][0] + pic1.shape[1], kp2[p][1]),random_color, 1, cv2.LINE_AA)
            cv2.circle(match_img, (kp1[i][0], kp1[i][1]), 2, (0, 255, 255), -1)
            cv2.circle(match_img, (kp2[p][0] + pic1.shape[1], kp2[p][1]), 2, (0, 255, 255), -1)
    cv2.imwrite(str(name)+".jpg",match_img)



def draw_lines_ignore_mask(pic1, pic2, kp1, kp2, mask):
    kp1 = kp1.astype(np.int)
    kp2 = kp2.astype(np.int)
    match_img = np.hstack([pic1, pic2])
    for i, p in enumerate(mask):
        if p == 1:
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(match_img, (kp1[i][0][0], kp1[i][0][1]), (kp2[i][0][0] + pic1.shape[1], kp2[i][0][1]),random_color, 1, cv2.LINE_AA)
            cv2.circle(match_img, (kp1[i][0][0], kp1[i][0][1]), 2, (0, 255, 255), -1)
            cv2.circle(match_img, (kp2[i][0][0] + pic1.shape[1], kp2[i][0][1]), 2, (0, 255, 255), -1)
    # cv2.imwrite(str(name)+".jpg",match_img)
    return match_img

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        
        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis

def showMath(files:list, method:str):
    imageA= cv2.imread(files[0])
    imageB = cv2.imread(files[1])

    imageA[:, 0:int(imageA.shape[1]*0.75)] = (0,0,0)
    imageB[:, int(imageB.shape[1]*0.25):] = (0,0,0)
    #  转换为gray
    gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 查找关键点和描述符
    (kps1, features1) = detectAndDescribe(gray1, method)
    (kps2, features2) = detectAndDescribe(gray2, method)
    
        

    
    # 创建BFMatcher 对象    
    bf = createMatcher(method,True)

    if method == 'ORB' or method == 'BRISK' or method == 'SURF' or method == 'SIFT':
        matches = bf.match(features1, features2)
        matches = sorted(matches, key = lambda x:x.distance)
        src_pts = np.float32([kps1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)  # 测试图像的关键点的坐标位置
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)  # 样本图像的关键点的坐标位置
        (H, status) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        img3 = draw_lines_ignore_mask(imageA, imageB, src_pts, dst_pts, status)
        # img3 = cv2.drawMatches(imageA,kps1,imageB,kps2,matches[:30], NULL,flags=2)
        # cv2.imwrite('./match.png', img3)
    return img3
    
def SIFT(img):
    I =  cv2.imread(img)
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(I, None)

    cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imwrite('sift_keypoints.jpg',I)
    return I

# def SURF(img):
#     I = cv2.imread(img)
#     descriptor = cv2.SURF_create()
#     (kps, features) = descriptor.detectAndCompute(I, None)
#     cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     # cv2.imwrite('surf_keypoints.jpg',I)
#     return I

def BRISK(img):
    I = cv2.imread(img)
    descriptor = cv2.BRISK_create()
    (kps, features) = descriptor.detectAndCompute(I, None)
    cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('brisk_keypoints.jpg',I)
    return I

def ORB(img):
    I = cv2.imread(img)
    descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(I, None)
    cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('orb_keypoints.jpg',I)
    return I


if __name__ == '__main__':
    img = './1.JPG'
    SIFT(img)
    # SURF(img)
    BRISK(img)
    ORB(img)


# if __name__ == '__main__':
#     print("hello, world")
#     showMath(['./0_left.png', './2_right.png'], 'SIFT')





class runMatchImage(QThread):
    def __init__(self, imageDir, algoType) -> None:
        super(runMatchImage, self).__init__()
        self.dir = imageDir
        self.algoType = algoType

    def run(self):
        print(self.dir)
        BASEDIR = self.dir
        OUTDIRBase = os.path.join(BASEDIR, 'out')
        leftFileName = ''
        rightFileName = ''
        for root , dirs, files in os.walk(BASEDIR):
            if files is not NULL:
                for file in files:
                    if file.startswith('0_'):
                        leftFileName = os.path.join(root, file)
                    elif file.startswith('2_'):
                        rightFileName = os.path.join(root, file)
                    
                    if os.path.exists(leftFileName) and os.path.exists(rightFileName):
                        dir = root.replace(BASEDIR,'')
                        if dir != '\out':
                            outfilepath = OUTDIRBase + dir
                
                        if not os.path.exists(outfilepath):
                            os.makedirs(outfilepath)
                        outfileName = os.path.join(outfilepath, self.algoType + "_Match.png")
                        if self.algoType != 'MODEL':
                            print('=====')
                            matchImg = showMath([leftFileName, rightFileName], self.algoType)
                            cv2.imwrite(outfileName, matchImg)
                            leftFileName = ''
                            rightFileName = ''
                        else:
                            matchImg = showMath_Model([leftFileName, rightFileName], outfileName)
                        







    