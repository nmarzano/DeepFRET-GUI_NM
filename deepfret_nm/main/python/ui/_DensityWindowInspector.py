# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/main/python/ui/DensityWindowInspector.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DensityWindowInspector(object):
    def setupUi(self, DensityWindowInspector):
        DensityWindowInspector.setObjectName("DensityWindowInspector")
        DensityWindowInspector.resize(281, 159)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            DensityWindowInspector.sizePolicy().hasHeightForWidth()
        )
        DensityWindowInspector.setSizePolicy(sizePolicy)
        DensityWindowInspector.setMinimumSize(QtCore.QSize(0, 0))
        DensityWindowInspector.setMaximumSize(QtCore.QSize(1000, 1000))
        self.gridLayout = QtWidgets.QGridLayout(DensityWindowInspector)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.overlayLabel = QtWidgets.QLabel(DensityWindowInspector)
        self.overlayLabel.setObjectName("overlayLabel")
        self.gridLayout_2.addWidget(self.overlayLabel, 4, 0, 1, 1)
        self.colorLabel = QtWidgets.QLabel(DensityWindowInspector)
        self.colorLabel.setObjectName("colorLabel")
        self.gridLayout_2.addWidget(self.colorLabel, 2, 0, 1, 1)
        self.smoothingLabel = QtWidgets.QLabel(DensityWindowInspector)
        self.smoothingLabel.setObjectName("smoothingLabel")
        self.gridLayout_2.addWidget(self.smoothingLabel, 0, 0, 1, 1)
        self.resolutionLabel = QtWidgets.QLabel(DensityWindowInspector)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.gridLayout_2.addWidget(self.resolutionLabel, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(DensityWindowInspector)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 3, 0, 1, 1)
        self.resolutionSlider = QtWidgets.QSlider(DensityWindowInspector)
        self.resolutionSlider.setMinimum(10)
        self.resolutionSlider.setMaximum(100)
        self.resolutionSlider.setProperty("value", 50)
        self.resolutionSlider.setTracking(False)
        self.resolutionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.resolutionSlider.setInvertedControls(False)
        self.resolutionSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.resolutionSlider.setTickInterval(1000)
        self.resolutionSlider.setObjectName("resolutionSlider")
        self.gridLayout_2.addWidget(self.resolutionSlider, 1, 1, 1, 1)
        self.smoothingSlider = QtWidgets.QSlider(DensityWindowInspector)
        self.smoothingSlider.setMinimum(1)
        self.smoothingSlider.setMaximum(20)
        self.smoothingSlider.setProperty("value", 10)
        self.smoothingSlider.setTracking(False)
        self.smoothingSlider.setOrientation(QtCore.Qt.Horizontal)
        self.smoothingSlider.setInvertedControls(False)
        self.smoothingSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.smoothingSlider.setTickInterval(1000)
        self.smoothingSlider.setObjectName("smoothingSlider")
        self.gridLayout_2.addWidget(self.smoothingSlider, 0, 1, 1, 1)
        self.pointAlphaSlider = QtWidgets.QSlider(DensityWindowInspector)
        self.pointAlphaSlider.setMinimum(0)
        self.pointAlphaSlider.setMaximum(20)
        self.pointAlphaSlider.setProperty("value", 10)
        self.pointAlphaSlider.setTracking(False)
        self.pointAlphaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.pointAlphaSlider.setInvertedControls(False)
        self.pointAlphaSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.pointAlphaSlider.setTickInterval(1000)
        self.pointAlphaSlider.setObjectName("pointAlphaSlider")
        self.gridLayout_2.addWidget(self.pointAlphaSlider, 3, 1, 1, 1)
        self.colorSlider = QtWidgets.QSlider(DensityWindowInspector)
        self.colorSlider.setMinimum(1)
        self.colorSlider.setMaximum(20)
        self.colorSlider.setProperty("value", 3)
        self.colorSlider.setTracking(False)
        self.colorSlider.setOrientation(QtCore.Qt.Horizontal)
        self.colorSlider.setInvertedControls(False)
        self.colorSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.colorSlider.setTickInterval(1000)
        self.colorSlider.setObjectName("colorSlider")
        self.gridLayout_2.addWidget(self.colorSlider, 2, 1, 1, 1)
        self.overlayCheckBox = QtWidgets.QCheckBox(DensityWindowInspector)
        self.overlayCheckBox.setText("")
        self.overlayCheckBox.setObjectName("overlayCheckBox")
        self.gridLayout_2.addWidget(self.overlayCheckBox, 4, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 4, 0, 1, 1)

        self.retranslateUi(DensityWindowInspector)
        QtCore.QMetaObject.connectSlotsByName(DensityWindowInspector)

    def retranslateUi(self, DensityWindowInspector):
        _translate = QtCore.QCoreApplication.translate
        DensityWindowInspector.setWindowTitle(
            _translate("DensityWindowInspector", "Adjust")
        )
        self.overlayLabel.setText(
            _translate("DensityWindowInspector", "Overlay Points:")
        )
        self.colorLabel.setText(_translate("DensityWindowInspector", "Colors:"))
        self.smoothingLabel.setText(
            _translate("DensityWindowInspector", "Smoothing: ")
        )
        self.resolutionLabel.setText(
            _translate("DensityWindowInspector", "Resolution: ")
        )
        self.label.setText(
            _translate("DensityWindowInspector", "Overlay Alpha:")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    DensityWindowInspector = QtWidgets.QDialog()
    ui = Ui_DensityWindowInspector()
    src.main.python.ui.setupUi(DensityWindowInspector)
    DensityWindowInspector.show()
    sys.exit(app.exec_())
