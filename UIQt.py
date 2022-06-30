import this
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from Iimageprocess import Process


class UI_Image(QWidget):
    def __init__(self, parent=None):
        super(UI_Image, self).__init__(parent)
        self.initUI()
        self.process = Process()
        self.srcImagePath = ""

    def initUI(self):
        self.setGeometry(300, 300, 365, 280)
        self.setWindowTitle("test")
        self.openButton = QPushButton("OpenFile")
        self.openButton.clicked.connect(lambda: self.onButtonClick(self.openButton))

        self.leftlist = QListWidget()
        self.leftlist.insertItem(0, "滤波")
        self.leftlist.insertItem(1, "Demosic")
        self.leftlist.insertItem(2, "直方图")

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

    def filterUI(self):
        self.filterTypelayout = QFormLayout()
        self.typeLabel = QLabel("type:")
        self.combox = QComboBox()
        self.combox.setObjectName("滤波")
        self.combox.addItems(["None", "mean", "Gauss", "box", "2D", "median", "bil"])
        self.filterTypelayout.addRow(self.typeLabel, self.combox)
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

    def onButtonClick(self, btn):
        # print("the button is ", btn.text())
        if btn.text() == "cal":
            if len(self.srcImagePath.strip()) > 0:
                # imageInfo = (self.srcImagePath, self.combox.currentText(), self.wightLineEdit.text(), self.highLineEdit.text(), self.bitcombox.currentText())
                imageInfo = {
                    "namePath": self.srcImagePath,
                    "typeCal": self.combox.currentText(),
                    "width": self.wightLineEdit.text(),
                    "height": self.highLineEdit.text(),
                    "bit": self.bitcombox.currentText(),
                    "pattern": self.patternComBox.currentText(),
                    "blackLevel": self.blacklevelLineEdit.text(),
                }
                str = self.process.imageprocess(imageInfo)
                self.dstImageLab.setPixmap(QPixmap(str))
            else:
                print("str is None")

        if btn.text() == "OpenFile":
            self.srcImagePath, _ = QFileDialog.getOpenFileName(
                self, "Open file", QDir.currentPath(), "Image files (*.jpg *.gif *.raw)"
            )
            self.srcImageLab.setPixmap(QPixmap(self.srcImagePath))

    def setImagePorcess(self, imgprocess):
        self.process = imgprocess
