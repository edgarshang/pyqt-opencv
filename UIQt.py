import logging as log
import this
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from numpy import histogram

from yaml import warnings
from Iimageprocess import Process, Show
from imageProcessThread import imageProcessThread


# 设置⽇志等级和输出⽇志格式
log.basicConfig(level=log.DEBUG,
format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
log.debug('这是⼀个debug级别的⽇志信息')



class UI_Image(QWidget, Show):
    def __init__(self, parent=None):
        super(UI_Image, self).__init__(parent)
        self.initUI()
        self.setWindowIcon(QIcon("./icon.png"))
        self.process = Process()
        self.yolov5Handler = Process()
        self.yolov8Handler = Process()
        self.landmarkHandler = Process()
        self.ageGentleHandler = Process()
        self.srcImagePath = ""
        self.imageProcessThread = None
        self.defaultSetting()

    
        

    def initUI(self):
        self.setWindowTitle("PyQt Pytorch-CV")
        self.openButton = QPushButton("OpenFile")
        self.openButton.clicked.connect(lambda: self.onButtonClick(self.openButton))

        self.calButton = QPushButton("Run")
        self.calButton.clicked.connect(lambda: self.onButtonClick(self.calButton))

        self.chooseFolderButton = QPushButton("ImageFolder...")
        self.chooseFolderButton.clicked.connect(lambda: self.onButtonClick(self.chooseFolderButton))

        # self.chooseFilesButton = QPushButton("ChooseFiles...")
        # self.chooseFilesButton.clicked.connect(lambda: self.onButtonClick(self.chooseFilesButton))

        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, "YOLOv5")
        self.leftlist.insertItem(1, "YOLOv8")
        self.leftlist.insertItem(2, "FasterRcnn")
        self.leftlist.insertItem(3, "MaskRcnn")
        self.leftlist.insertItem(4, "Unet")
        self.leftlist.insertItem(5, "LandMark")
        self.leftlist.insertItem(6, "AgeGentle")
        


        self.stacklayout = QHBoxLayout()

        self.Stack = QStackedWidget(self)
        self.stacklayout.addWidget(self.Stack)
        self.qstatckGroupBox = QGroupBox("ChooseType")
        self.qstatckGroupBox.setLayout(self.stacklayout)
        # self.Stack.setStyleSheet()

        self.stackfilter = QWidget()
        self.Stack.addWidget(self.stackfilter)
        self.filterUI(self.stackfilter)

        # self.statckDemosic = QWidget()
        # self.Stack.addWidget(self.statckDemosic)
        # self.rawDemosicUI()

        # self.statckCannyDetect = QWidget()
        # self.Stack.addWidget(self.statckCannyDetect)
        # self.CannyedgeDetectionUI()

        # self.stackThresholding = QWidget()
        # self.Stack.addWidget(self.stackThresholding)
        # self.ThresholdingUI()

        # self.stackGeomTrans = QWidget()
        # self.Stack.addWidget(self.stackGeomTrans)
        # self.GromTransFormUI(self.stackGeomTrans)

        self.leftlist.currentRowChanged.connect(self.display)

        

        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.openButton)
        self.vlayout.addWidget(self.calButton)
        self.vlayout.addWidget(self.chooseFolderButton)
        
        # self.vlayout.addWidget(self.chooseFilesButton)
        self.vlayout.addStretch()

        self.showTabWidget = QTabWidget()

        self.showImageWidget = QWidget()
        self.showTabWidget.addTab(self.showImageWidget, "showResult")
        self.showImageWidgetUI(self.showImageWidget)

        # self.showHistogramWidget = QWidget()
        # self.showTabWidget.addTab(self.showHistogramWidget, "Histogram")
        # self.showHistogramWidgetUI(self.showHistogramWidget)

        # self.showMeanCurveWidget = QWidget()
        # self.showTabWidget.addTab(self.showMeanCurveWidget, "MeanCurve")
        # self.showMeanCurveWidgetUI(self.showMeanCurveWidget)

        # self.showstitchWidget = QWidget()
        # self.showTabWidget.addTab(self.showstitchWidget, "stitchResult")
        # self.showstitchWidgetUI(self.showMeanCurveWidget)

        self.showTabLayout = QHBoxLayout()

        self.showTabLayout.addWidget(self.showTabWidget)

        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.leftlist)
        self.hlayout.addWidget(self.qstatckGroupBox)
        self.hlayout.addLayout(self.vlayout)
        self.hlayout.addStretch()
        self.hlayout.addLayout(self.showTabLayout)
        self.hlayout.addStretch()

        self.setLayout(self.hlayout)

    def display(self, i):
        self.Stack.setCurrentIndex(i)

    def showMeanCurveWidgetUI(self, uilayout):
        pass

    def showstitchWidgetUI(self, uilayout):
        pass

    def showHistogramWidgetUI(self, uilayout):
        self.showHistogramHlayout = QHBoxLayout()
        self.showHistogramLabel = QLabel()
        self.showHistogramLabel.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.showHistogramLabel.setScaledContents(True)
        self.showHistogramHlayout.addWidget(self.showHistogramLabel)

        uilayout.setLayout(self.showHistogramHlayout)

    def showImageWidgetUI(self, uilayout):
        self.imagelabLayout = QHBoxLayout()

        self.srcImageLab = QLabel()
        self.srcImageLab.setMinimumSize(640, 640)
        self.srcImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.srcImageLab.setScaledContents(True)
        # self.dstImageLab = QLabel()
        # self.dstImageLab.setMinimumSize(320, 240)
        # self.dstImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        # self.dstImageLab.setScaledContents(True)
        self.imagelabLayout.addWidget(self.srcImageLab)
        # self.imagelabLayout.addWidget(self.dstImageLab)

        uilayout.setLayout(self.imagelabLayout)
    
    def imageStitchingUI(self, uilayout):
        hlayout = QVBoxLayout()
        groupbox = QGroupBox("algoType")
        siftradioButton = QRadioButton("SIFT")
        orbradioButton = QRadioButton("ORB")
        brisktradioButton = QRadioButton("BRISK")
        modelradioButton = QRadioButton("MODEL")
        modelradioButton.setCheckable(False)
        hlayout.addWidget(siftradioButton)
        hlayout.addWidget(orbradioButton)
        hlayout.addWidget(brisktradioButton)
        hlayout.addWidget(modelradioButton)
        groupbox.setLayout(hlayout)

        siftradioButton.toggled.connect(
            lambda: self.onRadioButtonToggled(siftradioButton)
        )
        

        orbradioButton.toggled.connect(
            lambda: self.onRadioButtonToggled(orbradioButton)
        )

        brisktradioButton.toggled.connect(
            lambda: self.onRadioButtonToggled(brisktradioButton)
        )

        self.keyPointCheckoutBox = QCheckBox("关键点显示")
        self.imageStitchCheckoutBox = QCheckBox("拼接")
        self.saveStitchCheckoutBox = QCheckBox("保存")

        self.stitchInfo = {"Algo": "SIFT"}
        siftradioButton.setChecked(True)

        hlayoutStitch = QVBoxLayout()
        hlayoutStitch.addWidget(groupbox)
        hlayoutStitch.addWidget(self.keyPointCheckoutBox)
        hlayoutStitch.addWidget(self.imageStitchCheckoutBox)
        hlayoutStitch.addWidget(self.saveStitchCheckoutBox)
        # 

        uilayout.setLayout(hlayoutStitch)

    def defaultSetting(self):
        self.settings = QSettings("config.ini", QSettings.IniFormat)
        
        self.pathLineEdit.setText(self.settings.value("SETTING/ImagePath"))

        self.srcImagePath = self.pathLineEdit.text()
        # log.debug(f'srcImagepaht is {self.srcImagePath}')
        # log.debug(len(self.srcImagePath.strip()))

        # ui.lineEditIPAddr.setText(settings.value("SERVER/server_ip"))
        # ui.lineEditPort.setText(settings.value("SERVER/server_port"))                       


  

    def filterUI(self, uiLayout):
        self.filterTypelayout = QFormLayout()
        self.typeLabel = QLabel("type:")
        self.combox = QComboBox()
        self.combox.addItems(["Pt", "onnx","fruit_self"])
        self.filterTypelayout.addRow(self.typeLabel, self.combox)

        self.pathLabel = QLabel("Path: ")
        self.pathLineEdit = QLineEdit()
        self.pathLineEdit.setReadOnly(True)
        self.filterTypelayout.addRow(self.pathLabel, self.pathLineEdit)

        self.deployLabel = QLabel("Deploy Mode: ")
        self.deployComboBox = QComboBox()
        self.deployComboBox.addItems(["Openvino", "onnxruntime","tensorrt", "opencv"])
        self.filterTypelayout.addRow(self.deployLabel, self.deployComboBox)

        self.cameraLabel = QLabel("Camera :")
        self.cameraCheckBox = QCheckBox()
        self.filterTypelayout.addRow(self.cameraLabel, self.cameraCheckBox)

        uiLayout.setLayout(self.filterTypelayout)

    def imageshow(self, info):
        self.showMatImage(info)


    def showMatImage(self, mat):
        dst = QImage(mat.data, mat.shape[1], mat.shape[0], mat.shape[1]*mat.shape[2], QImage.Format.Format_RGB888)
        # showLabel.setPixmap(QPixmap(str))
        self.srcImageLab.setPixmap(QPixmap.fromImage(dst.rgbSwapped()))


    def showImage(self, showLabel, str):
        print("....")
        import cv2
        src = cv2.imread(str)
        # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        print(src.shape)
        dst = QImage(src.data, src.shape[1], src.shape[0], src.shape[1]*src.shape[2], QImage.Format.Format_RGB888)
        # showLabel.setPixmap(QPixmap(str))
        showLabel.setPixmap(QPixmap.fromImage(dst.rgbSwapped()))
        # showLabel.setScaledContents(True)

    def DemosicHandle(self):
        if len(self.srcImagePath.strip()) > 0:
            imageInfo = {
                "funcType": self.leftlist.currentItem().text(),
                "namePath": self.srcImagePath,
                "typeCal": "Demosic",
                "width": self.wightLineEdit.text(),
                "height": self.highLineEdit.text(),
                "bit": self.bitcombox.currentText(),
                "pattern": self.patternComBox.currentText(),
                "blackLevel": self.blacklevelLineEdit.text(),
            }
            str, histogram_path = self.process.imageprocess(imageInfo)
            # self.dstImageLab.setPixmap(QPixmap(str))
            self.showImage(self.dstImageLab, str)
            self.showImage(self.showHistogramLabel, histogram_path)

    def filterHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": self.combox.currentText(),
            "kernelSize": self.kernelSizeCombox.currentText(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        # self.dstImageLab.setPixmap(QPixmap(str))
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def cannyHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": "Canny",
            "the1": self.CannythresholdEdit1.text(),
            "the2": self.CannythresholdEdit2.text(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def thresholdHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": "Threshold",
            "typeThreshold": self.thresholdCombox.currentText(),
            "thresholdValue": self.thresholdEditVal.text(),
            "threshodlMaxValue": self.thresholdEditMaxVal.text(),
            "thresAdmethod": self.thresStatusInfo["method"],
            "thresAdType": self.thresStatusInfo["thresholdType"],
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def geomTransformHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": "GeomTransform",
            "typeGeom": self.gromTransFormCombox.currentText(),
            "geomDx": self.gromTransFormDxEdit.text(),
            "geomDy": self.gromTransFormDyEdit.text(),
            "rotate": self.gromTransCombox.currentText(),
            "gemoPanx": self.gromTransPanMoveX.text(),
            "gemoPany": self.gromTransPanMoveY.text(),
            "Angle": self.geomTransFormAngle.text(),
            "Scale": self.geomTransFormScale.text(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)
    
    def MorphologyHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": "Morphology",
            "typeMorph": self.MorphologyCombox.currentText(),
            "KernelMor": self.MorphologyKernelCombox.currentText(),
            "countMor":self.MorphologyCountSpinBox.text(),
        }

        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)
        # print("hell")

    def ImageGradientHandle(self):
        imageInfo = {
            "funcType" : self.leftlist.currentItem().text(),
            "namePath" : self.srcImagePath,
            "typeCal" : "ImageGradient",
            "typeGrad" : self.imageGradientCombox.currentText(),
            "X_DIR":self.imageGradientCheckBoxX.isChecked(),
            "Y_DIR":self.imageGradientCheckBoxY.isChecked(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def ImagePyramidHandle(self):
        imageInfo = {
            "funcType":self.leftlist.currentItem().text(),
            "namePath":self.srcImagePath,
            "typeCal":"ImagePyramid",
            "typePyramid":self.imagePyramidCombox.currentText(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def ImageHistHandle(self):
        imageInfo = {
            "funcType":self.leftlist.currentItem().text(),
            "namePath":self.srcImagePath,
            "typeCal":"ImageHist",
            "typeHist":self.imageHistCombox.currentText(),
            "BinsHist" : self.imageHistBins.text(),
        }

        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def imageFourTransHandle(self):
        imageInfo = {
            "funcType":self.leftlist.currentItem().text(),
            "namePath":self.srcImagePath,
            "typeCal":"ImageFourTrans",
            "TypeFour":self.imageFourTransCombox.currentText(),
            "FrequencySize":self.imageFourTransLineEdit.text(),
        }
        str, histogram_path = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)
        self.showImage(self.showHistogramLabel, histogram_path)

    def imageStitchHandle(self):
        imageInfo = {
            "funcType":self.leftlist.currentItem().text(),
            "leftnamePath":self.srcImagePaths[0],
            "rightnamePath":self.srcImagePaths[1],
            "typeCal":"ImageStitch",
            "algoType":self.stitchInfo["Algo"],
        }
        left, right = self.process.imageprocess(imageInfo)
        self.showImage(self.srcImageLab, left)
        self.showImage(self.dstImageLab, right)
        log.debug("this is imageStitchHandle")




    def onRadioButtonToggled(self, btn):
        if btn.isChecked() == True:
            print("btn.text()", btn.text())
            if btn.text() == "高斯":
                self.thresStatusInfo["method"] = "高斯"
            elif btn.text() == "邻阈像素点":
                self.thresStatusInfo["method"] = "邻阈像素点"
            elif btn.text() == "BINARY_INV":
                self.thresStatusInfo["thresholdType"] = "BINARY_INV"
            elif btn.text() == "BINARY":
                self.thresStatusInfo["thresholdType"] = "BINARY"
            elif btn.text() == "SIFT":
                log.debug("sift")
                self.stitchInfo["Algo"] = "SIFT"
            elif btn.text() == "BRISK":
                log.debug("brisk")
                self.stitchInfo["Algo"] = "BRISK"
            elif btn.text() == "ORB":
                log.debug("orb")
                self.stitchInfo["Algo"] = "ORB"

    def onButtonClick(self, btn):
        if btn.text() == "Run":
            log.debug(self.leftlist.currentItem().text())
            log.debug(len(self.srcImagePath.strip()))
            log.debug(self.srcImagePath)
            if len(self.srcImagePath.strip()) > 0:
                if self.leftlist.currentItem().text() == "YOLOv5":
                    log.info("this is the yolov5 test")

                    imageInfo = {
                    "namePath": self.srcImagePath,
                    "typeCal": self.combox.currentText(),
                    }

                    # Done 放到线程中处理，释放前台
                    self.imageProcessThread = imageProcessThread()
                    self.imageProcessThread.setHandlerAndPath(self.yolov5Handler.imageprocess, imageInfo)
                    self.imageProcessThread.start()
                elif self.leftlist.currentItem().text() == "YOLOv8":
                    log.debug("this is the yolov8 test")
                    imageInfo = {
                    "namePath": self.srcImagePath,
                    "typeCal": self.combox.currentText(),
                    "deploy": self.deployComboBox.currentText(),
                    }
                    self.imageProcessThread = imageProcessThread()
                    self.imageProcessThread.setHandlerAndPath(self.yolov8Handler.imageprocess, imageInfo)
                    self.imageProcessThread.start()
                elif self.leftlist.currentItem().text() == "LandMark":
                    log.debug(f'land mark')
                    imageInfo = {
                    "namePath": self.srcImagePath,
                    "camera":self.cameraCheckBox.isChecked()
                    }
                    self.imageProcessThread = imageProcessThread()
                    self.imageProcessThread.setHandlerAndPath(self.landmarkHandler.imageprocess, imageInfo)
                    self.imageProcessThread.start()
                elif self.leftlist.currentItem().text() == "AgeGentle":
                    log.debug(f'AgeGentle')
                    imageInfo = {
                        "namePath": self.srcImagePath,
                        "camera":self.cameraCheckBox.isChecked()
                    }
                    self.imageProcessThread = imageProcessThread()
                    self.imageProcessThread.setHandlerAndPath(self.ageGentleHandler.imageprocess, imageInfo)
                    self.imageProcessThread.start()

            else:
                log.error("str is None")
        elif btn.text() == "OpenFile":
            self.srcImagePath, _ = QFileDialog.getOpenFileName(
                self,
                "Open file",
                QDir.currentPath(),
                "Image files (*.jpg *.gif *.png *.mp4 *.jpeg)",
            )
            self.pathLineEdit.setText(self.srcImagePath)
            self.settings.setValue("SETTING/ImagePath", self.srcImagePath)
            # print(_)
            
        elif btn.text() == "ImageFolder...":
            print("choose the test Folder")
            self.srcImagePath = QFileDialog.getExistingDirectory(self, "choose the folder...", QDir.currentPath())
            self.pathLineEdit.setText(self.srcImagePath)
            self.settings.setValue("SETTING/ImagePath", self.srcImagePath)
            # if not self.srcImagePath.endswith(
            #     ".raw"
            # ) and not self.srcImagePath.endswith(".bin"):
            #     self.showImage(self.srcImageLab, self.srcImagePath)

        # if btn.text()  == "ChooseFiles...":
        #     self.srcImagePaths, _ = QFileDialog.getOpenFileNames(
        #         self,
        #         "Open files",
        #         QDir.currentPath(),
        #         "Image files (*.jpg *.gif *.png)",
        #     )
        #     if len(self.srcImagePaths) == 2:
        #         self.showImage(self.srcImageLab, self.srcImagePaths[0])
        #         self.showImage(self.dstImageLab, self.srcImagePaths[1])


    def setImagePorcess(self, imgprocess):
        self.process = imgprocess

    def setYoloV5Process(self, yolov5Handle):
        self.yolov5Handler = yolov5Handle

    def setYoloV8Process(self, yolov5Handle):
        self.yolov8Handler = yolov5Handle
    
    def setLandMarkProcess(self, landMarkHandle):
        self.landmarkHandler = landMarkHandle

    def setAgeGentleProcess(self, ageGentleHandle):
        self.ageGentleHandler = ageGentleHandle

    def closeEvent(self, a0: QCloseEvent) -> None:
        print("close it")
        self.imageProcessThread.stop()
        return super().closeEvent(a0)