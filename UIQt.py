from logging import warning
import this
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

from yaml import warnings
from Iimageprocess import Process


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

        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, "滤波")
        self.leftlist.insertItem(1, "Demosic")
        self.leftlist.insertItem(2, "Canny")
        self.leftlist.insertItem(3, "阈值处理")

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

        self.leftlist.currentRowChanged.connect(self.display)

        self.calButton = QPushButton("cal")
        self.calButton.clicked.connect(lambda: self.onButtonClick(self.calButton))

        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.openButton)
        self.vlayout.addWidget(self.calButton)
        self.vlayout.addStretch()

        self.srcImageLab = QLabel()
        self.srcImageLab.setMinimumSize(320, 240)
        self.srcImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.srcImageLab.setScaledContents(True)
        self.dstImageLab = QLabel()
        self.dstImageLab.setMinimumSize(320, 240)
        self.dstImageLab.setStyleSheet("QLabel{background-color:rgb(0,0,0)}")
        self.dstImageLab.setScaledContents(True)

        self.imagelabLayout = QHBoxLayout()
        # self.imagelabLayout.addStretch()
        self.imagelabLayout.addWidget(self.srcImageLab)
        self.imagelabLayout.addWidget(self.dstImageLab)
        # self.imagelabLayout.addStretch()

        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.leftlist)
        self.hlayout.addWidget(self.qstatckGroupBox)
        self.hlayout.addLayout(self.vlayout)
        self.hlayout.addStretch()
        self.hlayout.addLayout(self.imagelabLayout)
        self.hlayout.addStretch()

        self.setLayout(self.hlayout)

    def display(self, i):
        self.Stack.setCurrentIndex(i)

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
            str = self.process.imageprocess(imageInfo)
            # self.dstImageLab.setPixmap(QPixmap(str))
            self.showImage(self.dstImageLab, str)

    def filterHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": self.combox.currentText(),
            "kernelSize" : self.kernelSizeCombox.currentText(),
        }
        str = self.process.imageprocess(imageInfo)
        # self.dstImageLab.setPixmap(QPixmap(str))
        self.showImage(self.dstImageLab, str)

    def cannyHandle(self):
        imageInfo = {
            "funcType": self.leftlist.currentItem().text(),
            "namePath": self.srcImagePath,
            "typeCal": "Canny",
            "the1": self.CannythresholdEdit1.text(),
            "the2": self.CannythresholdEdit2.text(),
        }
        str = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)

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
        str = self.process.imageprocess(imageInfo)
        self.showImage(self.dstImageLab, str)

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

    def onButtonClick(self, btn):
        if btn.text() == "cal":
            print(self.leftlist.currentItem().text())
            if len(self.srcImagePath.strip()) > 0:
                if self.leftlist.currentItem().text() == "Demosic":
                    self.DemosicHandle()
                elif self.leftlist.currentItem().text() == "滤波":
                    self.filterHandle()
                elif self.leftlist.currentItem().text() == "Canny":
                    self.cannyHandle()
                elif self.leftlist.currentItem().text() == "阈值处理":
                    self.thresholdHandle()
            else:
                print("str is None")

        if btn.text() == "OpenFile":
            self.srcImagePath, _ = QFileDialog.getOpenFileName(
                self,
                "Open file",
                QDir.currentPath(),
                "Image files (*.jpg *.gif *.raw *.bin)",
            )
            if not self.srcImagePath.endswith(
                ".raw"
            ) and not self.srcImagePath.endswith(".bin"):
                self.showImage(self.srcImageLab, self.srcImagePath)

    def setImagePorcess(self, imgprocess):
        self.process = imgprocess
