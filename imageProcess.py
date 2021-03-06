from genericpath import exists
import pstats
from cv2 import imread
from Iimageprocess import Process
import cv2
import numpy as np
import os
from matplotlib import cm, pyplot as plt


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
        plt.cla()
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
        if typeThreshold == "?????????":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY
            )
        elif typeThreshold == "????????????":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_BINARY_INV
            )
        elif typeThreshold == "????????????":
            t, r = cv2.threshold(o, thresholdValue, thresholdValueMax, cv2.THRESH_TRUNC)
        elif typeThreshold == "????????????":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO_INV
            )
        elif typeThreshold == "????????????":
            t, r = cv2.threshold(
                o, thresholdValue, thresholdValueMax, cv2.THRESH_TOZERO
            )
        elif typeThreshold == "???????????????":
            if imageInfo["thresAdmethod"] == "??????":
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
            elif imageInfo["thresAdmethod"] == "???????????????":
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
        if typeGeom == "??????":
            r = cv2.resize(
                o,
                None,
                fx=np.double(imageInfo["geomDx"]),
                fy=np.double(imageInfo["geomDy"]),
            )
        elif typeGeom == "??????":
            r = cv2.flip(o, int(imageInfo["rotate"]))
        elif typeGeom == "??????":
            h, w = o.shape[:2]
            M = np.float32(
                [[1, 0, int(imageInfo["gemoPanx"])], [0, 1, int(imageInfo["gemoPany"])]]
            )
            r = cv2.warpAffine(o, M, (w, h))
        elif typeGeom == "??????":
            h, w = o.shape[:2]
            M = cv2.getRotationMatrix2D(
                (w / 2, h / 2), int(imageInfo["Angle"]), np.double(imageInfo["Scale"])
            )
            r = cv2.warpAffine(o, M, (w, h))
        elif typeGeom == "??????":
            pass
        elif typeGeom == "?????????":
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
        if typeMorphology == "??????":
            r = cv2.erode(o, kernel, iterations=count)
        elif typeMorphology == "??????":
            r = cv2.dilate(o, kernel, iterations=count)
        elif typeMorphology == "?????????":
            r = cv2.morphologyEx(o, cv2.MORPH_OPEN, kernel, iterations=count)
        elif typeMorphology == "?????????":
            r = cv2.morphologyEx(o, cv2.MORPH_CLOSE, kernel, iterations=count)
        elif typeMorphology == "????????????":
            r = cv2.morphologyEx(o, cv2.MORPH_TOPHAT, kernel, iterations=count)
        elif typeMorphology == "????????????":
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
        if typeImagePyramid == "???????????????":
            r = cv2.pyrUp(o)
        elif typeImagePyramid == "???????????????":
            r = cv2.pyrDown(o)
        elif typeImagePyramid == "?????????????????????":
            G0 = o
            G1 = cv2.pyrDown(G0)
            r = G0 - cv2.pyrUp(G1)
        cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def imageHistFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath)

        typeImageHist = imageInfo["typeHist"]
        binsHist = int(imageInfo["BinsHist"])

        if typeImageHist == "numpy":
            plt.hist(o.ravel(), binsHist)
            if os.path.exists(dstname):
                os.remove(dstname)
            plt.savefig(dstname)
        elif typeImageHist == "CV2":
            histb = cv2.calcHist([o], [0],None, [binsHist], [0,256])
            plt.plot(histb, "b")
            plt.xlim([0, 256])

            # plt.plot(histb,'b')
            plt.savefig(dstname)
        elif "???????????????" == typeImageHist:
            o = imread(srcNamePath,cv2.IMREAD_GRAYSCALE)
            r = cv2.equalizeHist(o)
            cv2.imwrite(dstname, r)
        return dstname, self.calHistogram(srcNamePath)

    def imageFourTransFunc(self, imageInfo):
        filePath, filename = os.path.split(imageInfo["namePath"])
        srcNamePath = imageInfo["namePath"]
        Type = imageInfo["typeCal"]
        dstname, filetype = os.path.splitext(filename)
        dstname += "_" + Type + filetype
        dstname = os.path.join(filePath, dstname)
        print(f"dstname = {dstname}, filetype = {filetype}")
        o = imread(srcNamePath, 0)
        
        TypeFourTrans = imageInfo["TypeFour"]
        Size = int(imageInfo["FrequencySize"])

        if TypeFourTrans == "???????????????":
            f = np.fft.fft2(o)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            plt.subplot(121)
            plt.imshow(o, cmap='gray')
            plt.title('original')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('result')
            plt.axis('off')
            plt.savefig(dstname)
            plt.cla()
        elif TypeFourTrans == "??????????????????":
            f = np.fft.fft2(o)
            fshift = np.fft.fftshift(f)
            ifshift = np.fft.ifftshift(fshift)
            iimg = np.fft.ifft2(ifshift)
            iimg = np.abs(iimg)
            plt.subplot(121),plt.imshow(o, cmap='gray')
            plt.title('original'),plt.axis('off')
            plt.subplot(122), plt.imshow(iimg, cmap='gray')
            plt.title('iimg'), plt.axis('off')
            plt.savefig(dstname)
            plt.cla()
        elif TypeFourTrans == "????????????":
            f = np.fft.fft2(o)
            fshift = np.fft.fftshift(f)
            rows, cols = o.shape
            crow, ccol = int(rows/2), int(cols/2)
            fshift[crow - Size : crow + Size, ccol - Size : ccol + Size] = 0
            ishift = np.fft.ifftshift(fshift)
            iimg = np.fft.ifft2(ishift)
            iimg = np.abs(iimg)
            cv2.imwrite(dstname, iimg)
        elif TypeFourTrans == "????????????":
            f = np.fft.fft2(o)
            fshift = np.fft.fftshift(f)
            rows, cols = o.shape
            crow, ccol = int(rows/2), int(cols/2)
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - Size : crow + Size, ccol - Size : ccol + Size] = 1
            ifshift = fshift * mask
            iifshift = np.fft.ifftshift(ifshift)
            iimg = np.fft.ifft2(iifshift)
            iimg = np.abs(iimg)
            cv2.imwrite(dstname, iimg)
        elif TypeFourTrans == "cv???????????????":
            dft = cv2.dft(np.float32(o), flags=cv2.DFT_COMPLEX_OUTPUT)
            dftShift = np.fft.fftshift(dft)
            result = 20*np.log(cv2.magnitude(dftShift[:,:,0], dftShift[:,:,1]))
            plt.subplot(121), plt.imshow(o, cmap='gray')
            plt.title('original'),plt.axis('off')
            plt.subplot(122),plt.imshow(result, cmap='gray')
            plt.title('result'), plt.axis('off')
            plt.savefig(dstname)
            plt.cla()
        elif TypeFourTrans == "cv??????????????????":
            dft = cv2.dft(np.float32(o), flags=cv2.DFT_COMPLEX_OUTPUT)
            dftShift = np.fft.fftshift(dft)
            ishift = np.fft.ifftshift(dftShift)
            iImg = cv2.idft(ishift)
            # iImg = cv2.magnitude(iImg[:,:,0], iImg[:,;,1])
            iImg = cv2.magnitude(iImg[:,:,0], iImg[:,:,1])
            plt.subplot(121), plt.imshow(o, cmap='gray')
            plt.title('original'), plt.axis('off')
            plt.subplot(122), plt.imshow(iImg, cmap='gray')
            plt.title('inverse'), plt.axis('off')
            plt.savefig(dstname)
            plt.cla()
        elif TypeFourTrans == "cv????????????":
            dft = cv2.dft(np.float32(o), flags=cv2.DFT_COMPLEX_OUTPUT)
            dftShift = np.fft.fftshift(dft)
            rows, cols = o.shape
            crow, ccol = int(rows/2), int(cols/2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow-Size:crow+Size, ccol-Size:ccol+Size] = 1
            fShift = dftShift*mask
            ishift = np.fft.ifftshift(fShift)
            iImg = cv2.idft(ishift)
            iImg = cv2.magnitude(iImg[:,:,0], iImg[:,:,1])
            plt.subplot(121), plt.imshow(o, cmap='gray')
            plt.title('original'), plt.axis('off')
            plt.subplot(122), plt.imshow(iImg, cmap='gray')
            plt.title('invers'), plt.axis('off')
            plt.savefig(dstname)
            plt.cla()
        return dstname, self.calHistogram(srcNamePath)
        

        
        

    def imageprocess(self, imageInfo):
        functionType = imageInfo["funcType"]
        print("In imageprocess...")

        if functionType == "Demosic":
            return self.demosicFunc(imageInfo)
        elif functionType == "??????":
            return self.burFunc(imageInfo)
        elif functionType == "Canny":
            return self.cannyFunc(imageInfo)
        elif functionType == "????????????":
            return self.thresFunc(imageInfo)
        elif functionType == "????????????":
            return self.geomtransFunc(imageInfo)
        elif functionType == "???????????????":
            return self.MorphologyFunc(imageInfo)
        elif functionType == "????????????":
            return self.imageGradientFunc(imageInfo)
        elif functionType == "???????????????":
            return self.imagePyramidFunc(imageInfo)
        elif functionType == "???????????????":
            return self.imageHistFunc(imageInfo)
        elif functionType == "???????????????":
            return self.imageFourTransFunc(imageInfo)

