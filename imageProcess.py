from cv2 import imread
from Iimageprocess import Process
import cv2
import numpy as np
import os


class opencvImage(Process):

    # def __init__(self, parent=None):
    #     super(opencvImage, self).__init__(parent)

    # def imageprocess(self,str,type="None"):

    def burFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        burType = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + burType + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath)

        if burType == "None":
            return srcNamePath

        elif burType == "mean":
            r = cv2.blur(o, (5, 5))
            cv2.imwrite(dstname, r)
            return dstname

        elif burType == "Gauss":
            r = cv2.GaussianBlur(o, (5, 5), 0, 0)
            cv2.imwrite(dstname, r)
            return dstname

        elif burType == "box":
            r = cv2.boxFilter(o, -1, (5, 5))
            cv2.imwrite(dstname, r)
            return dstname

        elif burType == "median":
            r = cv2.medianBlur(o, 3)
            cv2.imwrite(dstname, r)
            return dstname

        elif burType == "bil":
            r = cv2.bilateralFilter(o, 25, 100, 100)
            cv2.imwrite(dstname, r)
            return dstname

        elif burType == "2D":
            kernel = np.ones((9, 9), dtype=np.float32) / 81
            r = cv2.filter2D(o, -1, kernel)
            cv2.imwrite(dstname, r)
            return dstname

    def demosicFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        burType = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + burType + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        rawImage = np.fromfile(open(imageInfo["namePath"]), dtype="uint16")
        dstname = dstname.replace(".raw", ".jpg")
        rawImage = rawImage.reshape(
            (int(imageInfo["height"]), int(imageInfo["width"]))
        ).astype(np.float32)
        print(rawImage.shape)
        rawImage = rawImage - int(imageInfo["blackLevel"])
        rawbayer = rawImage.clip(0, 2 ** int(imageInfo["bit"]) - 1)

        rawbayer = rawbayer.astype(np.uint16)
        if imageInfo["pattern"] == "RGGB":
            self.bgrImage = cv2.cvtColor(rawbayer, cv2.COLOR_BAYER_RG2RGB)
        elif imageInfo["pattern"] == "BGGR":
            self.bgrImage = cv2.cvtColor(rawbayer, cv2.COLOR_BAYER_BG2RGB)
        elif imageInfo["pattern"] == "GRBG":
            self.bgrImage = cv2.cvtColor(rawbayer, cv2.COLOR_BAYER_GR2RGB)
        elif imageInfo["pattern"] == "GBRG":
            self.bgrImage = cv2.cvtColor(rawbayer, cv2.COLOR_BAYER_GB2RGB)

        self.bgrImage = self.bgrImage >> (int(imageInfo["bit"]) - 8)
        self.bgrImage = self.bgrImage.astype(np.uint8)
        wbgain = np.array(
            [
                self.bgrImage[:, :, 0].mean() / self.bgrImage[:, :, :1].mean(),
                1.0,
                self.bgrImage[:, :, 2].mean() / self.bgrImage[:, :, :1].mean(),
            ],
            np.float32,
        )
        self.bgrImage = (self.bgrImage * wbgain / 256) ** (1 / 2.2) * 255 // 1

        cv2.imwrite(dstname, self.bgrImage)
        return dstname

    def cannyFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath)
        # thres1 = int()
        r = cv2.Canny(o, int(imageInfo["the1"]), int(imageInfo["the2"]))
        # r = cv2.Canny(o, 32, 128)
        cv2.imwrite(dstname, r)
        return dstname

    def thresFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath, 0)
        typeThreshold = imageInfo["typeThreshold"]
        thresholdValue = int(imageInfo["thresholdValue"])
        thresholdValueMax = int(imageInfo["threshodlMaxValue"])
        if typeThreshold == "二值化":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY)
        elif typeThreshold == "反二值化":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY_INV)
        elif typeThreshold == "截断阈值":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_TRUNC)
        elif typeThreshold == "超阈值零":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO_INV)
        elif typeThreshold == "低阈值零":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO)
        elif typeThreshold == "自适应阈值":
            if imageInfo["thresAdmethod"] == "高斯":  
                if imageInfo["thresAdType"] == "BINARY":
                    r = cv2.adaptiveThreshold(o, thresholdValueMax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
                elif imageInfo["thresAdType"] == "BINARY_INV":
                    r = cv2.adaptiveThreshold(o, thresholdValueMax, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
            elif imageInfo["thresAdmethod"] == "邻阈像素点":
                if imageInfo["thresAdType"] == "BINARY":
                    r = cv2.adaptiveThreshold(o, thresholdValueMax, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
                elif imageInfo["thresAdType"] == "BINARY_INV":
                    r = cv2.adaptiveThreshold(o, thresholdValueMax, cv2.ADAPTIVE_THRESH_MEAN_Cs, cv2.THRESH_BINARY_INV, 5, 3)
        elif typeThreshold == "Otsu":
            t, r = cv2.threshold(o, 0, thresholdValueMax, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite(dstname, r)
        return dstname


    def imageprocess(self, imageInfo):
        functionType = imageInfo["funcType"]

        if functionType == "Demosic":
            return self.demosicFunc(imageInfo)
        elif functionType == "滤波":
            return self.burFunc(imageInfo)
        elif functionType == "Canny":
            return self.cannyFunc(imageInfo)
        elif functionType == "阈值处理":
            return self.thresFunc(imageInfo)
