import logging as log
import this
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from numpy import histogram

from yaml import warnings
from Iimageprocess import Process

# 设置⽇志等级和输出⽇志格式
log.basicConfig(level=log.DEBUG,
format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
log.debug('这是⼀个debug级别的⽇志信息')



class UI_Image(QWidget):
    def __init__(self, parent=None):
        super(UI_Image, self).__init__(parent)
        self.initUI()
        self.setWindowIcon(QIcon("./icon.png"))
        self.process = Process()
        self.srcImagePath = ""

    def initUI(self):
        self.setWindowTitle("PyQt OpenCV")
        self.openButton = QPushButton("OpenFile")
        self.openButton.clicked.connect(lambda: self.onButtonClick(self.openButton))

        self.chooseFilesButton = QPushButton("ChooseFiles...")
        self.chooseFilesButton.clicked.connect(lambda: self.onButtonClick(self.chooseFilesButton))

        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, "滤波")
        self.leftlist.insertItem(1, "Demosic")
        self.leftlist.insertItem(2, "Canny")
        self.leftlist.insertItem(3, "阈值处理")
        self.leftlist.insertItem(4, "几何变换")
        self.leftlist.insertItem(5, "形态学操作")
        self.leftlist.insertItem(6, "图像梯度")
        self.leftlist.insertItem(7, "图像金字塔")
        self.leftlist.insertItem(8, "直方图处理")
        self.leftlist.insertItem(9, "傅里叶变换")
        self.leftlist.insertItem(10, "图像拼接")
        self.leftlist.insertItem(11, "人脸识别")
        self.leftlist.insertItem(12, "霍夫变换")
        self.leftlist.insertItem(13, "图像分割与提取")
        self.leftlist.insertItem(14, "视频处理")
        self.leftlist.insertItem(15, "绘图及交互")
        self.leftlist.insertItem(16, "K 近邻算法")
        self.leftlist.insertItem(17, "支持向量机")
        self.leftlist.insertItem(18, "K 均值聚类")
        self.leftlist.insertItem(19, "图像轮廓")
        self.leftlist.insertItem(20, "模板匹配")



        self.stacklayout = QHBoxLayout()

        self.Stack = QStackedWidget(self)
        self.stacklayout.addWidget(self.Stack)
        self.qstatckGroupBox = QGroupBox("ChooseType")
        self.qstatckGroupBox.setLayout(self.stacklayout)
        # self.Stack.setStyleSheet()

        self.stackfilter = QWidget()
        self.Stack.addWidget(self.stackfilter)
        self.filterUI()

        self.statckDemosic = QWidget()
        self.Stack.addWidget(self.statckDemosic)
        self.rawDemosicUI()

        self.statckCannyDetect = QWidget()
        self.Stack.addWidget(self.statckCannyDetect)
        self.CannyedgeDetectionUI()

        self.stackThresholding = QWidget()
        self.Stack.addWidget(self.stackThresholding)
        self.ThresholdingUI()

        self.stackGeomTrans = QWidget()
        self.Stack.addWidget(self.stackGeomTrans)
        self.GromTransFormUI(self.stackGeomTrans)

        self.morphology = QWidget()
        self.Stack.addWidget(self.morphology)
        self.MorphologyUI(self.morphology)

        self.imageGradient = QWidget()
        self.Stack.addWidget(self.imageGradient)
        self.imageGradientUI(self.imageGradient)

        self.imagePyramid = QWidget()
        self.Stack.addWidget(self.imagePyramid)
        self.imagePyramidUI(self.imagePyramid)

        self.imageHist = QWidget()
        self.Stack.addWidget(self.imageHist)
        self.imageHistUI(self.imageHist)

        self.imageFourTrans = QWidget()
        self.Stack.addWidget(self.imageFourTrans)
        self.imageFourTransUI(self.imageFourTrans)

        self.imageStitching = QWidget()
        self.Stack.addWidget(self.imageStitching)
        self.imageStitchingUI(self.imageStitching)

        self.leftlist.currentRowChanged.connect(self.display)

        self.calButton = QPushButton("cal")
        self.calButton.clicked.connect(lambda: self.onButtonClick(self.calButton))

        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.openButton)
        self.vlayout.addWidget(self.calButton)
        self.vlayout.addWidget(self.chooseFilesButton)
        self.vlayout.addStretch()

        self.showTabWidget = QTabWidget()

        self.showImageWidget = QWidget()
        self.showTabWidget.addTab(self.showImageWidget, "image")
        self.showImageWidgetUI(self.showImageWidget)

        self.showHistogramWidget = QWidget()
        self.showTabWidget.addTab(self.showHistogramWidget, "Histogram")
        self.showHistogramWidgetUI(self.showHistogramWidget)

        self.showMeanCurveWidget = QWidget()
        self.showTabWidget.addTab(self.showMeanCurveWidget, "MeanCurve")
        self.showMeanCurveWidgetUI(self.showMeanCurveWidget)

        self.showstitchWidget = QWidget()
        self.showTabWidget.addTab(self.showstitchWidget, "stitchResult")
        self.showstitchWidgetUI(self.showMeanCurveWidget)

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
        self.srcImageLab.setMinimumSize(320, 240)
        self.srcImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.srcImageLab.setScaledContents(True)
        self.dstImageLab = QLabel()
        self.dstImageLab.setMinimumSize(320, 240)
        self.dstImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.dstImageLab.setScaledContents(True)
        self.imagelabLayout.addWidget(self.srcImageLab)
        self.imagelabLayout.addWidget(self.dstImageLab)

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


    def imageFourTransUI(self, uilayout):
        self.imageFourTransFormLayout = QFormLayout()
        self.imageFourTransCombox = QComboBox()
        self.imageFourTransCombox.addItems(["傅里叶变换", "逆傅里叶变换", "高通滤波", "低通滤波", "cv傅里叶变换", "cv逆傅里叶变换", "cv低通滤波"])
        self.imageFourTransFormLayout.addRow("Type:", self.imageFourTransCombox)
        self.imageFourTransLineEdit = QLineEdit("30")
        self.imageFourTransFormLayout.addRow("frequency:", self.imageFourTransLineEdit)

        uilayout.setLayout(self.imageFourTransFormLayout)
        

    def imageHistUI(self, uilayout):
        self.imageHistFormLayout = QFormLayout()
        self.imageHistCombox = QComboBox()

        self.imageHistCombox.addItems(["numpy", "CV2", "直方图均衡"])
        self.imageHistFormLayout.addRow("TypeHis:", self.imageHistCombox)

        self.imageHistBins = QLineEdit("256")
        self.imageHistFormLayout.addRow("Bins:", self.imageHistBins)
        uilayout.setLayout(self.imageHistFormLayout)


    def imagePyramidUI(self, uilayout):
        self.imagePyramidFormLayout = QFormLayout()
        self.imagePyramidCombox = QComboBox()

        self.imagePyramidCombox.addItems(["高斯上采样", "高斯下采样", "拉普拉斯金字塔"])
        self.imagePyramidFormLayout.addRow("采样：", self.imagePyramidCombox)

        uilayout.setLayout(self.imagePyramidFormLayout)

    def imageGradientUI(self, uilayout):
        self.imageGradientFormLayout = QFormLayout()

        self.imageGradientCombox = QComboBox()
        self.imageGradientCombox.addItems(["Sobel", "Scharr", "Laplacian"])

        self.imageGradientCheckBoxX = QCheckBox("x 方向")
        self.imageGradientCheckBoxX.setChecked(True)
        self.imageGradientCheckBoxY = QCheckBox("y 方向")

        self.imageGradientFormLayout.addRow("算子:", self.imageGradientCombox)
        self.imageGradientFormLayout.addRow("X:", self.imageGradientCheckBoxX)
        self.imageGradientFormLayout.addRow("Y:", self.imageGradientCheckBoxY)

        uilayout.setLayout(self.imageGradientFormLayout)

    def MorphologyUI(self, uilayout):
        self.morphologyFormLayout = QFormLayout()

        self.MorphologyCombox = QComboBox()
        self.MorphologyCombox.addItems(["腐蚀", "膨胀", "开运算", "闭运算", "礼帽运算", "黑帽运算"])
        self.morphologyFormLayout.addRow("MorphType:", self.MorphologyCombox)
        self.MorphologyKernelCombox = QComboBox()
        self.MorphologyKernelCombox.addItems(["3", "5", "7", "9", "11", "13"])
        self.morphologyFormLayout.addRow("Kernel:", self.MorphologyKernelCombox)
        self.MorphologyCountSpinBox = QSpinBox()
        self.MorphologyCountSpinBox.setMaximum(3)
        self.morphologyFormLayout.addRow("Count:", self.MorphologyCountSpinBox)

        uilayout.setLayout(self.morphologyFormLayout)

    def GromTransFormUI(self, uiLayout):
        self.gromTransFormLayout = QFormLayout()
        self.gromTransFormCombox = QComboBox()
        self.gromTransFormCombox.addItems(["缩放", "翻转", "访射", "旋转", "透视", "重映射"])
        self.gromTransFormLayout.addRow("tranType:", self.gromTransFormCombox)
        self.gromTransFormDxEdit = QLineEdit("1")
        self.gromTransFormDyEdit = QLineEdit("1")

        self.gromTransFormLayout.addRow("fx:", self.gromTransFormDxEdit)
        self.gromTransFormLayout.addRow("fy:", self.gromTransFormDyEdit)

        self.gromTransCombox = QComboBox()
        self.gromTransCombox.addItems(["0", "1", "-1"])
        self.gromTransCombox.setCurrentText("0")
        self.gromTransFormLayout.addRow("rotate:", self.gromTransCombox)

        self.gromTransPanMoveX = QLineEdit("100")
        self.gromTransPanMoveY = QLineEdit("100")

        self.gromTransFormLayout.addRow("PanX:", self.gromTransPanMoveX)
        self.gromTransFormLayout.addRow("PanX:", self.gromTransPanMoveY)

        self.geomTransFormAngle = QLineEdit("45")
        self.geomTransFormScale = QLineEdit("0.5")

        self.gromTransFormLayout.addRow("angle:", self.geomTransFormAngle)
        self.gromTransFormLayout.addRow("scale:", self.geomTransFormScale)

        uiLayout.setLayout(self.gromTransFormLayout)

    def ThresholdingUI(self):
        self.thresholdFormLayout = QFormLayout()
        self.thresholdLabel = QLabel("阈值类型:")
        self.thresholdCombox = QComboBox()
        self.thresholdCombox.setObjectName("阈值")
        self.thresholdCombox.addItems(
            ["二值化", "反二值化", "截断阈值", "超阈值零", "低阈值零", "自适应阈值", "Otsu"]
        )
        self.thresholdLabelVal = QLabel("阈值:")
        self.thresholdEditVal = QLineEdit("127")
        self.thresholdLabelMaxVal = QLabel("MaxVal:")
        self.thresholdEditMaxVal = QLineEdit("255")
        self.thresholdFormLayout.addRow(self.thresholdLabel, self.thresholdCombox)
        self.thresholdFormLayout.addRow(self.thresholdLabelVal, self.thresholdEditVal)
        self.thresholdFormLayout.addRow(
            self.thresholdLabelMaxVal, self.thresholdEditMaxVal
        )
        self.thresholdMethodGroupBox = QGroupBox("自适应方法")

        self.thresholdNibRadiobutton = QRadioButton("邻阈像素点")

        self.thresholdNibRadiobutton.toggled.connect(
            lambda: self.onRadioButtonToggled(self.thresholdNibRadiobutton)
        )

        self.thresholdGusRadiobutton = QRadioButton("高斯")
        self.thresholdGusRadiobutton.setChecked(True)

        self.thresholdGusRadiobutton.toggled.connect(
            lambda: self.onRadioButtonToggled(self.thresholdGusRadiobutton)
        )

        self.thresholdMethodVBoxLayout = QVBoxLayout()
        self.thresholdMethodVBoxLayout.addWidget(self.thresholdNibRadiobutton)
        self.thresholdMethodVBoxLayout.addWidget(self.thresholdGusRadiobutton)
        self.thresholdMethodGroupBox.setLayout(self.thresholdMethodVBoxLayout)

        self.thresholdTypeGroupBox = QGroupBox("thresholdType")

        self.thresholdBinRadiobutton = QRadioButton("BINARY")
        self.thresholdBinRadiobutton.setChecked(True)
        self.thresStatusInfo = {"method": "高斯", "thresholdType": "BINARY"}

        self.thresholdBinRadiobutton.toggled.connect(
            lambda: self.onRadioButtonToggled(self.thresholdBinRadiobutton)
        )

        self.thresholdBinInvRadiobutton = QRadioButton("BINARY_INV")

        self.thresholdBinInvRadiobutton.toggled.connect(
            lambda: self.onRadioButtonToggled(self.thresholdBinInvRadiobutton)
        )

        self.thresholdTypeVBoxLayout = QVBoxLayout()
        self.thresholdTypeVBoxLayout.addWidget(self.thresholdBinRadiobutton)
        self.thresholdTypeVBoxLayout.addWidget(self.thresholdBinInvRadiobutton)
        self.thresholdTypeGroupBox.setLayout(self.thresholdTypeVBoxLayout)

        self.thresholdFormLayout.addRow("自适应方法", self.thresholdMethodGroupBox)
        self.thresholdFormLayout.addRow("阈值处理方式", self.thresholdTypeGroupBox)

        # pIntValidator = QIntValidator(self)
        # pIntValidator.setRange(1, 99)
        pIntValidator = QIntValidator(self)
        pIntValidator.setRange(1, 254)
        self.thresholdEditVal.setValidator(pIntValidator)
        self.thresholdEditMaxVal.setValidator(pIntValidator)

        self.stackThresholding.setLayout(self.thresholdFormLayout)

    def CannyedgeDetectionUI(self):
        self.cannyFormLayout = QFormLayout()
        thresholdLabel1 = QLabel("threshold-1:")
        self.CannythresholdEdit1 = QLineEdit("32")
        thresholdLabel2 = QLabel("threshold-2:")
        self.CannythresholdEdit2 = QLineEdit("128")
        self.cannyFormLayout.addRow(thresholdLabel1, self.CannythresholdEdit1)
        self.cannyFormLayout.addRow(thresholdLabel2, self.CannythresholdEdit2)
        pIntValidator = QIntValidator(self)
        pIntValidator.setRange(1, 254)
        self.CannythresholdEdit1.setValidator(pIntValidator)
        self.CannythresholdEdit2.setValidator(pIntValidator)
        self.statckCannyDetect.setLayout(self.cannyFormLayout)

    def filterUI(self):
        self.filterTypelayout = QFormLayout()
        self.typeLabel = QLabel("type:")
        self.combox = QComboBox()
        self.combox.setObjectName("滤波")
        self.combox.addItems(["None", "mean", "Gauss", "box", "2D", "median", "bil"])
        self.filterTypelayout.addRow(self.typeLabel, self.combox)
        self.filterSlider = QSlider(Qt.Horizontal)
        self.filterSlider.setMinimum(3)
        self.filterSlider.setMaximum(11)
        self.filterSlider.setSingleStep(2)
        self.filterSlider.setValue(5)
        self.filterSlider.setTickPosition(QSlider.TicksBelow)
        self.filterSlider.setTickInterval(2)
        self.kernelSizeCombox = QComboBox()
        self.kernelSizeCombox.addItems(["3", "5", "7", "9", "11", "13"])
        self.kernelSizeCombox.setCurrentText("5")
        self.filterTypelayout.addRow("kernel", self.filterSlider)
        self.filterTypelayout.addRow("kernelSize", self.kernelSizeCombox)
        self.stackfilter.setLayout(self.filterTypelayout)

    def rawDemosicUI(self):
        self.typelayout = QFormLayout()
        self.wightLabel = QLabel("W:")
        self.wightLineEdit = QLineEdit()
        self.wightLineEdit.setText("2688")
        self.typelayout.addRow(self.wightLabel, self.wightLineEdit)

        self.highLabel = QLabel("H:")
        self.highLineEdit = QLineEdit()
        self.highLineEdit.setText("1520")
        self.typelayout.addRow(self.highLabel, self.highLineEdit)

        reg = QRegExp("[0-9]+$")
        pValidator = QRegExpValidator(self)
        pValidator.setRegExp(reg)

        self.wightLineEdit.setValidator(pValidator)
        self.highLineEdit.setValidator(pValidator)

        self.bitLabel = QLabel("Bit:")
        self.bitcombox = QComboBox()
        self.bitcombox.addItems(["16", "8", "12", "14"])
        self.typelayout.addRow(self.bitLabel, self.bitcombox)

        self.patternLabel = QLabel("Pattern:")
        self.patternComBox = QComboBox()
        self.patternComBox.addItems(["RGGB", "BGGR", "GRBG", "GBRG"])
        self.typelayout.addRow(self.patternLabel, self.patternComBox)

        self.blacklevelLabel = QLabel("BlackLevel:")
        self.blacklevelLineEdit = QLineEdit("4096")
        self.blacklevelLineEdit.setValidator(pValidator)
        self.typelayout.addRow(self.blacklevelLabel, self.blacklevelLineEdit)

        self.statckDemosic.setLayout(self.typelayout)

    def showImage(self, showLabel, str):
        showLabel.setPixmap(QPixmap(str))

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
        if btn.text() == "cal":
            print(self.leftlist.currentItem().text())
            if len(self.srcImagePath.strip()) > 0 or len(self.srcImagePaths) == 2:
                if self.leftlist.currentItem().text() == "Demosic":
                    self.DemosicHandle()
                elif self.leftlist.currentItem().text() == "滤波":
                    self.filterHandle()
                elif self.leftlist.currentItem().text() == "Canny":
                    self.cannyHandle()
                elif self.leftlist.currentItem().text() == "阈值处理":
                    self.thresholdHandle()
                elif self.leftlist.currentItem().text() == "几何变换":
                    self.geomTransformHandle()
                elif self.leftlist.currentItem().text() == "形态学操作":
                    self.MorphologyHandle()
                elif self.leftlist.currentItem().text() == "图像梯度":
                    self.ImageGradientHandle()
                elif self.leftlist.currentItem().text() == "图像金字塔":
                    self.ImagePyramidHandle()
                elif self.leftlist.currentItem().text() == "直方图处理":
                    self.ImageHistHandle()
                elif self.leftlist.currentItem().text() == "傅里叶变换":
                    self.imageFourTransHandle()
                elif self.leftlist.currentItem().text() == "图像拼接":
                    self.imageStitchHandle()
            else:
                log.error("str is None")

        if btn.text() == "OpenFile":
            self.srcImagePath, _ = QFileDialog.getOpenFileName(
                self,
                "Open file",
                QDir.currentPath(),
                "Image files (*.jpg *.gif *.png *.raw *.bin)",
            )
            if not self.srcImagePath.endswith(
                ".raw"
            ) and not self.srcImagePath.endswith(".bin"):
                self.showImage(self.srcImageLab, self.srcImagePath)

        if btn.text()  == "ChooseFiles...":
            self.srcImagePaths, _ = QFileDialog.getOpenFileNames(
                self,
                "Open files",
                QDir.currentPath(),
                "Image files (*.jpg *.gif *.png)",
            )
            if len(self.srcImagePaths) == 2:
                self.showImage(self.srcImageLab, self.srcImagePaths[0])
                self.showImage(self.dstImageLab, self.srcImagePaths[1])


    def setImagePorcess(self, imgprocess):
        self.process = imgprocess
