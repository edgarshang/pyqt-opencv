import pstats
from cv2 import imread
from Iimageprocess import Process
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


class opencvImage(Process):

    # def __init__(self, parent=None):
    #     super(opencvImage, self).__init__(parent)

    # def imageprocess(self,str,type="None"):

    def calHistogram(self, imagePath):
        img = cv2.imread(imagePath)
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        histDstName = os.path.split(imagePath)[0] + "_histgrom.png"
        plt.savefig(histDstName)
        return histDstName

    def burFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        burType = imageInfo["typeCal"]
        kernelSize = int(imageInfo["kernelSize"])
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + burType + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath)

        if burType == "None":
            return srcNamePath, self.calHistogram(srcNamePath)

        elif burType == "mean":
            r = cv2.blur(o, (kernelSize, kernelSize))
            cv2.imwrite(dstname, r)

        elif burType == "Gauss":
            r = cv2.GaussianBlur(o, (kernelSize, kernelSize), 0, 0)
            cv2.imwrite(dstname, r)

        elif burType == "box":
            r = cv2.boxFilter(o, -1, (kernelSize, kernelSize))
            cv2.imwrite(dstname, r)

        elif burType == "median":
            r = cv2.medianBlur(o, kernelSize)
            cv2.imwrite(dstname, r)

        elif burType == "bil":
            r = cv2.bilateralFilter(o, 25, 100, 100)
            cv2.imwrite(dstname, r)

        elif burType == "2D":
            kernel = np.ones((kernelSize, kernelSize), dtype=np.float32) / (
                kernelSize * kernelSize
            )
            r = cv2.filter2D(o, -1, kernel)
            cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def demosicFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        burType = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + burType + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        rawImage = np.fromfile(open(imageInfo["namePath"]), dtype="uint16")
        if dstname.endswith(".raw"):
            dstname = dstname.replace(".raw", ".jpg")
        else:
            dstname = dstname.replace(".bin", ".jpg")
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
        return dstname, self.calHistogram(dstname)

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
        return dstname, self.calHistogram(srcNamePath)

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
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY
            )
        elif typeThreshold == "反二值化":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY_INV
            )
        elif typeThreshold == "截断阈值":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_TRUNC)
        elif typeThreshold == "超阈值零":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO_INV
            )
        elif typeThreshold == "低阈值零":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO
            )
        elif typeThreshold == "自适应阈值":
            if imageInfo["thresAdmethod"] == "高斯":
                if imageInfo["thresAdType"] == "BINARY":
                    r = cv2.adaptiveThreshold(
                        o,
                        thresholdValueMax,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        5,
                        3,
                    )
                elif imageInfo["thresAdType"] == "BINARY_INV":
                    r = cv2.adaptiveThreshold(
                        o,
                        thresholdValueMax,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        5,
                        3,
                    )
            elif imageInfo["thresAdmethod"] == "邻阈像素点":
                if imageInfo["thresAdType"] == "BINARY":
                    r = cv2.adaptiveThreshold(
                        o,
                        thresholdValueMax,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        5,
                        3,
                    )
                elif imageInfo["thresAdType"] == "BINARY_INV":
                    r = cv2.adaptiveThreshold(
                        o,
                        thresholdValueMax,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY_INV,
                        5,
                        3,
                    )
        elif typeThreshold == "Otsu":
            t, r = cv2.threshold(
                o, 0, thresholdValueMax, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def geomtransFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath)
        typeGeom = imageInfo["typeGeom"]
        if typeGeom == "缩放":
            r = cv2.resize(
                o,
                None,
                fx=np.double(imageInfo["geomDx"]),
                fy=np.double(imageInfo["geomDy"]),
            )
        elif typeGeom == "翻转":
            r = cv2.flip(o, int(imageInfo["rotate"]))
        elif typeGeom == "访射":
            h, w = o.shape[:2]
            M = np.float32(
                [[1, 0, int(imageInfo["gemoPanx"])], [0, 1, int(imageInfo["gemoPany"])]]
            )
            r = cv2.warpAffine(o, M, (w, h))
        elif typeGeom == "旋转":
            h, w = o.shape[:2]
            M = cv2.getRotationMatrix2D(
                (w / 2, h / 2), int(imageInfo["Angle"]), np.double(imageInfo["Scale"])
            )
            r = cv2.warpAffine(o, M, (w, h))
        elif typeGeom == "透视":
            pass
        elif typeGeom == "重映射":
            pass
        cv2.imwrite(dstname, r)

        return dstname, self.calHistogram(srcNamePath)

    def MorphologyFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath, cv2.IMREAD_UNCHANGED)
        
        typeMorphology = imageInfo["typeMorph"]
        kernelSize = int(imageInfo["KernelMor"])
        count = int(imageInfo["countMor"])
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        if typeMorphology == "腐蚀":
            r = cv2.erode(o, kernel, iterations=count)
        elif typeMorphology == "膨胀":
            r = cv2.dilate(o, kernel, iterations=count)
        elif typeMorphology == "开运算":
            r = cv2.morphologyEx(o, cv2.MORPH_OPEN, kernel, iterations=count)
        elif typeMorphology == "闭运算":
            r = cv2.morphologyEx(o, cv2.MORPH_CLOSE, kernel, iterations=count)
        elif typeMorphology == "礼帽运算":
            r = cv2.morphologyEx(o, cv2.MORPH_TOPHAT, kernel, iterations=count)
        elif typeMorphology == "黑帽运算":
            r = cv2.morphologyEx(o, cv2.MORPH_BLACKHAT, kernel, iterations=count)
        cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def imageGradientFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath, cv2.IMREAD_GRAYSCALE)

        typeImageGrad = imageInfo["typeGrad"]
        xGrid = 1 if imageInfo["X_DIR"] == True else 0
        yGrid = 1 if imageInfo["Y_DIR"] == True else 0

        if typeImageGrad == "Sobel":
            r = cv2.Sobel(o, cv2.CV_64F, xGrid, yGrid)
            r = cv2.convertScaleAbs(r)
        elif typeImageGrad == "Scharr":
            r = cv2.Scharr(o, cv2.CV_64F, xGrid, yGrid)
            r = cv2.convertScaleAbs(r)
        elif typeImageGrad == "Laplacian":
            r = cv2.Laplacian(o, cv2.CV_64F)
            r = cv2.convertScaleAbs(r)
        cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def imagePyramidFunc(self, imageInfo):
        print("hello, world")
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath, cv2.IMREAD_GRAYSCALE)

        typeImagePyramid = imageInfo["typePyramid"]
        print(typeImagePyramid)
        if typeImagePyramid == "高斯上采样":
            r = cv2.pyrUp(o)
        elif typeImagePyramid == "高斯下采样":
            r = cv2.pyrDown(o)
        elif typeImagePyramid == "拉普拉斯金字塔":
            G0 = o
            G1 = cv2.pyrDown(G0)
            r = G0 - cv2.pyrUp(G1)
        cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

        
        

    def imageprocess(self, imageInfo):
        functionType = imageInfo["funcType"]
        print("In imageprocess...")

        if functionType == "Demosic":
            return self.demosicFunc(imageInfo)
        elif functionType == "滤波":
            return self.burFunc(imageInfo)
        elif functionType == "Canny":
            return self.cannyFunc(imageInfo)
        elif functionType == "阈值处理":
            return self.thresFunc(imageInfo)
        elif functionType == "几何变换":
            return self.geomtransFunc(imageInfo)
        elif functionType == "形态学操作":
            return self.MorphologyFunc(imageInfo)
        elif functionType == "图像梯度":
            return self.imageGradientFunc(imageInfo)
        elif functionType == "图像金字塔":
            return self.imagePyramidFunc(imageInfo)
