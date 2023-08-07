# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/main/python/ui/MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 615)
        MainWindow.setMinimumSize(QtCore.QSize(1200, 300))
        MainWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        MainWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        MainWindow.setWindowTitle("Images")
        MainWindow.setStatusTip("")
        MainWindow.setWhatsThis("")
        MainWindow.setAccessibleName("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        MainWindow.setDocumentMode(False)
        MainWindow.setDockNestingEnabled(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setStyleSheet("background rgb(0,0,0)")
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint
        )
        self.gridLayout.setHorizontalSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.contrastBoxHiGreen = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.contrastBoxHiGreen.sizePolicy().hasHeightForWidth()
        )
        self.contrastBoxHiGreen.setSizePolicy(sizePolicy)
        self.contrastBoxHiGreen.setMinimumSize(QtCore.QSize(52, 21))
        self.contrastBoxHiGreen.setFrame(True)
        self.contrastBoxHiGreen.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToNearestValue
        )
        self.contrastBoxHiGreen.setKeyboardTracking(False)
        self.contrastBoxHiGreen.setDecimals(0)
        self.contrastBoxHiGreen.setMaximum(100.0)
        self.contrastBoxHiGreen.setSingleStep(1.0)
        self.contrastBoxHiGreen.setProperty("value", 20.0)
        self.contrastBoxHiGreen.setObjectName("contrastBoxHiGreen")
        self.gridLayout.addWidget(self.contrastBoxHiGreen, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.gridLayout.addItem(spacerItem, 13, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            20,
            10,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.gridLayout.addItem(spacerItem1, 2, 0, 1, 1)
        self.labelRed = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelRed.sizePolicy().hasHeightForWidth()
        )
        self.labelRed.setSizePolicy(sizePolicy)
        self.labelRed.setObjectName("labelRed")
        self.gridLayout.addWidget(self.labelRed, 9, 0, 1, 1)
        self.labelContrastRed = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelContrastRed.sizePolicy().hasHeightForWidth()
        )
        self.labelContrastRed.setSizePolicy(sizePolicy)
        self.labelContrastRed.setObjectName("labelContrastRed")
        self.gridLayout.addWidget(self.labelContrastRed, 1, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            20,
            10,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.gridLayout.addItem(spacerItem2, 6, 0, 1, 1)
        self.labelColocGreenRedSpots = QtWidgets.QLabel(self.centralWidget)
        self.labelColocGreenRedSpots.setObjectName("labelColocGreenRedSpots")
        self.gridLayout.addWidget(self.labelColocGreenRedSpots, 11, 1, 1, 1)
        self.contrastBoxLoGreen = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.contrastBoxLoGreen.sizePolicy().hasHeightForWidth()
        )
        self.contrastBoxLoGreen.setSizePolicy(sizePolicy)
        self.contrastBoxLoGreen.setMinimumSize(QtCore.QSize(52, 21))
        self.contrastBoxLoGreen.setFrame(True)
        self.contrastBoxLoGreen.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToNearestValue
        )
        self.contrastBoxLoGreen.setKeyboardTracking(False)
        self.contrastBoxLoGreen.setDecimals(0)
        self.contrastBoxLoGreen.setMaximum(100.0)
        self.contrastBoxLoGreen.setSingleStep(1.0)
        self.contrastBoxLoGreen.setObjectName("contrastBoxLoGreen")
        self.gridLayout.addWidget(self.contrastBoxLoGreen, 0, 1, 1, 1)
        self.labelParticlesColoc = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelParticlesColoc.setFont(font)
        self.labelParticlesColoc.setObjectName("labelParticlesColoc")
        self.gridLayout.addWidget(self.labelParticlesColoc, 7, 0, 1, 1)
        self.spotsRedSpinBox = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.spotsRedSpinBox.sizePolicy().hasHeightForWidth()
        )
        self.spotsRedSpinBox.setSizePolicy(sizePolicy)
        self.spotsRedSpinBox.setStyleSheet("")
        self.spotsRedSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spotsRedSpinBox.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.spotsRedSpinBox.setKeyboardTracking(False)
        self.spotsRedSpinBox.setMaximum(9999)
        self.spotsRedSpinBox.setObjectName("spotsRedSpinBox")
        self.gridLayout.addWidget(self.spotsRedSpinBox, 5, 1, 1, 2)
        self.spotsGrnSpinBox = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.spotsGrnSpinBox.sizePolicy().hasHeightForWidth()
        )
        self.spotsGrnSpinBox.setSizePolicy(sizePolicy)
        self.spotsGrnSpinBox.setStyleSheet("")
        self.spotsGrnSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spotsGrnSpinBox.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.spotsGrnSpinBox.setKeyboardTracking(False)
        self.spotsGrnSpinBox.setMaximum(9999)
        self.spotsGrnSpinBox.setObjectName("spotsGrnSpinBox")
        self.gridLayout.addWidget(self.spotsGrnSpinBox, 4, 1, 1, 2)
        self.labelContrastGreen = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelContrastGreen.sizePolicy().hasHeightForWidth()
        )
        self.labelContrastGreen.setSizePolicy(sizePolicy)
        self.labelContrastGreen.setObjectName("labelContrastGreen")
        self.gridLayout.addWidget(self.labelContrastGreen, 0, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(
            20,
            10,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.gridLayout.addItem(spacerItem3, 12, 0, 1, 1)
        self.labelGreenVal = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelGreenVal.sizePolicy().hasHeightForWidth()
        )
        self.labelGreenVal.setSizePolicy(sizePolicy)
        self.labelGreenVal.setObjectName("labelGreenVal")
        self.gridLayout.addWidget(self.labelGreenVal, 4, 0, 1, 1)
        self.labelGreen = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelGreen.sizePolicy().hasHeightForWidth()
        )
        self.labelGreen.setSizePolicy(sizePolicy)
        self.labelGreen.setObjectName("labelGreen")
        self.gridLayout.addWidget(self.labelGreen, 8, 0, 1, 1)
        self.labelRedSpots = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelRedSpots.sizePolicy().hasHeightForWidth()
        )
        self.labelRedSpots.setSizePolicy(sizePolicy)
        self.labelRedSpots.setObjectName("labelRedSpots")
        self.gridLayout.addWidget(self.labelRedSpots, 9, 1, 1, 1)
        self.labelGreenSpots = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelGreenSpots.sizePolicy().hasHeightForWidth()
        )
        self.labelGreenSpots.setSizePolicy(sizePolicy)
        self.labelGreenSpots.setObjectName("labelGreenSpots")
        self.gridLayout.addWidget(self.labelGreenSpots, 8, 1, 1, 1)
        self.labelRedVal = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelRedVal.sizePolicy().hasHeightForWidth()
        )
        self.labelRedVal.setSizePolicy(sizePolicy)
        self.labelRedVal.setObjectName("labelRedVal")
        self.gridLayout.addWidget(self.labelRedVal, 5, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            20,
            10,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.gridLayout.addItem(spacerItem4, 10, 0, 1, 1)
        self.contrastBoxLoRed = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.contrastBoxLoRed.sizePolicy().hasHeightForWidth()
        )
        self.contrastBoxLoRed.setSizePolicy(sizePolicy)
        self.contrastBoxLoRed.setMinimumSize(QtCore.QSize(52, 21))
        self.contrastBoxLoRed.setFrame(True)
        self.contrastBoxLoRed.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToNearestValue
        )
        self.contrastBoxLoRed.setKeyboardTracking(False)
        self.contrastBoxLoRed.setDecimals(0)
        self.contrastBoxLoRed.setMaximum(100.0)
        self.contrastBoxLoRed.setSingleStep(1.0)
        self.contrastBoxLoRed.setObjectName("contrastBoxLoRed")
        self.gridLayout.addWidget(self.contrastBoxLoRed, 1, 1, 1, 1)
        self.labelColocGreenRed = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelColocGreenRed.sizePolicy().hasHeightForWidth()
        )
        self.labelColocGreenRed.setSizePolicy(sizePolicy)
        self.labelColocGreenRed.setObjectName("labelColocGreenRed")
        self.gridLayout.addWidget(self.labelColocGreenRed, 11, 0, 1, 1)
        self.labelLowerThres = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelLowerThres.setFont(font)
        self.labelLowerThres.setObjectName("labelLowerThres")
        self.gridLayout.addWidget(self.labelLowerThres, 3, 0, 1, 1)
        self.contrastBoxHiRed = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.contrastBoxHiRed.sizePolicy().hasHeightForWidth()
        )
        self.contrastBoxHiRed.setSizePolicy(sizePolicy)
        self.contrastBoxHiRed.setMinimumSize(QtCore.QSize(52, 21))
        self.contrastBoxHiRed.setFrame(True)
        self.contrastBoxHiRed.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToNearestValue
        )
        self.contrastBoxHiRed.setKeyboardTracking(False)
        self.contrastBoxHiRed.setDecimals(0)
        self.contrastBoxHiRed.setMaximum(100.0)
        self.contrastBoxHiRed.setSingleStep(1.0)
        self.contrastBoxHiRed.setProperty("value", 20.0)
        self.contrastBoxHiRed.setObjectName("contrastBoxHiRed")
        self.gridLayout.addWidget(self.contrastBoxHiRed, 1, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 1, 1)
        self.LayoutBox = QtWidgets.QHBoxLayout()
        self.LayoutBox.setSpacing(6)
        self.LayoutBox.setObjectName("LayoutBox")
        self.gridLayout_2.addLayout(self.LayoutBox, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.contrastBoxLoGreen, self.contrastBoxHiGreen)
        MainWindow.setTabOrder(self.contrastBoxHiGreen, self.contrastBoxLoRed)
        MainWindow.setTabOrder(self.contrastBoxLoRed, self.contrastBoxHiRed)
        MainWindow.setTabOrder(self.contrastBoxHiRed, self.spotsGrnSpinBox)
        MainWindow.setTabOrder(self.spotsGrnSpinBox, self.spotsRedSpinBox)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.labelRed.setText(_translate("MainWindow", "Red"))
        self.labelContrastRed.setText(_translate("MainWindow", "Red contrast"))
        self.labelColocGreenRedSpots.setText(
            _translate("MainWindow", "nGreenRedSpots")
        )
        self.labelParticlesColoc.setText(
            _translate("MainWindow", "Particles Colocalized")
        )
        self.labelContrastGreen.setText(
            _translate("MainWindow", "Green contrast")
        )
        self.labelGreenVal.setText(_translate("MainWindow", "Green channel"))
        self.labelGreen.setText(_translate("MainWindow", "Green"))
        self.labelRedSpots.setText(_translate("MainWindow", "nRedSpots"))
        self.labelGreenSpots.setText(_translate("MainWindow", "nGreenSpots"))
        self.labelRedVal.setText(_translate("MainWindow", "Red channel"))
        self.labelColocGreenRed.setText(_translate("MainWindow", "Green/Red"))
        self.labelLowerThres.setText(
            _translate("MainWindow", "Particle Finder")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    src.main.python.ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
