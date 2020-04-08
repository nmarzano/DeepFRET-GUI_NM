# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/SimulatorWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SimulatorWindow(object):
    def setupUi(self, SimulatorWindow):
        SimulatorWindow.setObjectName("SimulatorWindow")
        SimulatorWindow.resize(1200, 600)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            SimulatorWindow.sizePolicy().hasHeightForWidth()
        )
        SimulatorWindow.setSizePolicy(sizePolicy)
        SimulatorWindow.setMinimumSize(QtCore.QSize(1200, 600))
        SimulatorWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        SimulatorWindow.setBaseSize(QtCore.QSize(1200, 600))
        SimulatorWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        SimulatorWindow.setWindowTitle("smFRET Trace Simulator")
        SimulatorWindow.setStatusTip("")
        SimulatorWindow.setWhatsThis("")
        SimulatorWindow.setAccessibleName("")
        SimulatorWindow.setAutoFillBackground(False)
        SimulatorWindow.setStyleSheet("")
        SimulatorWindow.setDocumentMode(False)
        SimulatorWindow.setDockNestingEnabled(False)
        SimulatorWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(SimulatorWindow)
        self.centralWidget.setStyleSheet("background rgb(0,0,0)")
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mpl_LayoutBox = QtWidgets.QHBoxLayout()
        self.mpl_LayoutBox.setSpacing(6)
        self.mpl_LayoutBox.setObjectName("mpl_LayoutBox")
        self.gridLayout_2.addLayout(self.mpl_LayoutBox, 1, 2, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.gridLayout.setHorizontalSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.inputBlinkingProbability = QtWidgets.QDoubleSpinBox(
            self.centralWidget
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputBlinkingProbability.sizePolicy().hasHeightForWidth()
        )
        self.inputBlinkingProbability.setSizePolicy(sizePolicy)
        self.inputBlinkingProbability.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputBlinkingProbability.setStyleSheet("")
        self.inputBlinkingProbability.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputBlinkingProbability.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputBlinkingProbability.setKeyboardTracking(False)
        self.inputBlinkingProbability.setMaximum(1.0)
        self.inputBlinkingProbability.setObjectName("inputBlinkingProbability")
        self.gridLayout.addWidget(self.inputBlinkingProbability, 8, 1, 1, 1)
        self.labelScrambleProbability = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelScrambleProbability.sizePolicy().hasHeightForWidth()
        )
        self.labelScrambleProbability.setSizePolicy(sizePolicy)
        self.labelScrambleProbability.setObjectName("labelScrambleProbability")
        self.gridLayout.addWidget(self.labelScrambleProbability, 1, 0, 1, 1)
        self.inputTraceLength = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputTraceLength.sizePolicy().hasHeightForWidth()
        )
        self.inputTraceLength.setSizePolicy(sizePolicy)
        self.inputTraceLength.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputTraceLength.setStyleSheet("")
        self.inputTraceLength.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputTraceLength.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputTraceLength.setKeyboardTracking(False)
        self.inputTraceLength.setMinimum(10)
        self.inputTraceLength.setMaximum(1000)
        self.inputTraceLength.setProperty("value", 200)
        self.inputTraceLength.setObjectName("inputTraceLength")
        self.gridLayout.addWidget(self.inputTraceLength, 0, 1, 1, 1)
        self.inputNoiseLo = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputNoiseLo.sizePolicy().hasHeightForWidth()
        )
        self.inputNoiseLo.setSizePolicy(sizePolicy)
        self.inputNoiseLo.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputNoiseLo.setStyleSheet("")
        self.inputNoiseLo.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.inputNoiseLo.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputNoiseLo.setKeyboardTracking(False)
        self.inputNoiseLo.setProperty("value", 0.05)
        self.inputNoiseLo.setObjectName("inputNoiseLo")
        self.gridLayout.addWidget(self.inputNoiseLo, 10, 1, 1, 1)
        self.checkBoxScaler = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxScaler.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxScaler.setObjectName("checkBoxScaler")
        self.gridLayout.addWidget(self.checkBoxScaler, 13, 3, 1, 1)
        self.labelTransitionProbability = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelTransitionProbability.sizePolicy().hasHeightForWidth()
        )
        self.labelTransitionProbability.setSizePolicy(sizePolicy)
        self.labelTransitionProbability.setObjectName(
            "labelTransitionProbability"
        )
        self.gridLayout.addWidget(self.labelTransitionProbability, 9, 0, 1, 1)
        self.labelAAmismatch = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelAAmismatch.sizePolicy().hasHeightForWidth()
        )
        self.labelAAmismatch.setSizePolicy(sizePolicy)
        self.labelAAmismatch.setObjectName("labelAAmismatch")
        self.gridLayout.addWidget(self.labelAAmismatch, 11, 0, 1, 1)
        self.inputMismatchLo = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputMismatchLo.sizePolicy().hasHeightForWidth()
        )
        self.inputMismatchLo.setSizePolicy(sizePolicy)
        self.inputMismatchLo.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputMismatchLo.setStyleSheet("")
        self.inputMismatchLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputMismatchLo.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputMismatchLo.setKeyboardTracking(False)
        self.inputMismatchLo.setMinimum(-1.0)
        self.inputMismatchLo.setMaximum(1.0)
        self.inputMismatchLo.setProperty("value", -0.05)
        self.inputMismatchLo.setObjectName("inputMismatchLo")
        self.gridLayout.addWidget(self.inputMismatchLo, 11, 1, 1, 1)
        self.labelAggregateProbability = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelAggregateProbability.sizePolicy().hasHeightForWidth()
        )
        self.labelAggregateProbability.setSizePolicy(sizePolicy)
        self.labelAggregateProbability.setObjectName(
            "labelAggregateProbability"
        )
        self.gridLayout.addWidget(self.labelAggregateProbability, 2, 0, 1, 1)
        self.inputDonorMeanLifetime = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputDonorMeanLifetime.sizePolicy().hasHeightForWidth()
        )
        self.inputDonorMeanLifetime.setSizePolicy(sizePolicy)
        self.inputDonorMeanLifetime.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputDonorMeanLifetime.setStyleSheet("")
        self.inputDonorMeanLifetime.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputDonorMeanLifetime.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputDonorMeanLifetime.setKeyboardTracking(False)
        self.inputDonorMeanLifetime.setMinimum(1)
        self.inputDonorMeanLifetime.setMaximum(9999)
        self.inputDonorMeanLifetime.setProperty("value", 200)
        self.inputDonorMeanLifetime.setObjectName("inputDonorMeanLifetime")
        self.gridLayout.addWidget(self.inputDonorMeanLifetime, 6, 1, 1, 1)
        self.inputFretStateMeans = QtWidgets.QLineEdit(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputFretStateMeans.sizePolicy().hasHeightForWidth()
        )
        self.inputFretStateMeans.setSizePolicy(sizePolicy)
        self.inputFretStateMeans.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputFretStateMeans.setMouseTracking(False)
        self.inputFretStateMeans.setInputMethodHints(QtCore.Qt.ImhNone)
        self.inputFretStateMeans.setObjectName("inputFretStateMeans")
        self.gridLayout.addWidget(self.inputFretStateMeans, 4, 1, 1, 1)
        self.checkBoxBleedthrough = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxBleedthrough.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxBleedthrough.setObjectName("checkBoxBleedthrough")
        self.gridLayout.addWidget(self.checkBoxBleedthrough, 12, 3, 1, 1)
        self.inputNumberOfTraces = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputNumberOfTraces.sizePolicy().hasHeightForWidth()
        )
        self.inputNumberOfTraces.setSizePolicy(sizePolicy)
        self.inputNumberOfTraces.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputNumberOfTraces.setStyleSheet("")
        self.inputNumberOfTraces.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputNumberOfTraces.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputNumberOfTraces.setKeyboardTracking(False)
        self.inputNumberOfTraces.setMaximum(999999)
        self.inputNumberOfTraces.setProperty("value", 100)
        self.inputNumberOfTraces.setObjectName("inputNumberOfTraces")
        self.gridLayout.addWidget(self.inputNumberOfTraces, 16, 1, 1, 1)
        self.inputMismatchHi = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputMismatchHi.sizePolicy().hasHeightForWidth()
        )
        self.inputMismatchHi.setSizePolicy(sizePolicy)
        self.inputMismatchHi.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputMismatchHi.setStyleSheet("")
        self.inputMismatchHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputMismatchHi.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputMismatchHi.setKeyboardTracking(False)
        self.inputMismatchHi.setProperty("value", 0.05)
        self.inputMismatchHi.setObjectName("inputMismatchHi")
        self.gridLayout.addWidget(self.inputMismatchHi, 11, 2, 1, 1)
        self.labelDonorMeanLifetime = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelDonorMeanLifetime.sizePolicy().hasHeightForWidth()
        )
        self.labelDonorMeanLifetime.setSizePolicy(sizePolicy)
        self.labelDonorMeanLifetime.setObjectName("labelDonorMeanLifetime")
        self.gridLayout.addWidget(self.labelDonorMeanLifetime, 6, 0, 1, 1)
        self.labelNoise = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelNoise.sizePolicy().hasHeightForWidth()
        )
        self.labelNoise.setSizePolicy(sizePolicy)
        self.labelNoise.setObjectName("labelNoise")
        self.gridLayout.addWidget(self.labelNoise, 10, 0, 1, 1)
        self.inputAcceptorMeanLifetime = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputAcceptorMeanLifetime.sizePolicy().hasHeightForWidth()
        )
        self.inputAcceptorMeanLifetime.setSizePolicy(sizePolicy)
        self.inputAcceptorMeanLifetime.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputAcceptorMeanLifetime.setStyleSheet("")
        self.inputAcceptorMeanLifetime.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputAcceptorMeanLifetime.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputAcceptorMeanLifetime.setKeyboardTracking(False)
        self.inputAcceptorMeanLifetime.setMinimum(1)
        self.inputAcceptorMeanLifetime.setMaximum(9999)
        self.inputAcceptorMeanLifetime.setProperty("value", 200)
        self.inputAcceptorMeanLifetime.setObjectName(
            "inputAcceptorMeanLifetime"
        )
        self.gridLayout.addWidget(self.inputAcceptorMeanLifetime, 7, 1, 1, 1)
        self.checkBoxDlifetime = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxDlifetime.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxDlifetime.setObjectName("checkBoxDlifetime")
        self.gridLayout.addWidget(self.checkBoxDlifetime, 6, 3, 1, 1)
        self.checkBoxTransitionProbability = QtWidgets.QCheckBox(
            self.centralWidget
        )
        self.checkBoxTransitionProbability.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.checkBoxTransitionProbability.setObjectName(
            "checkBoxTransitionProbability"
        )
        self.gridLayout.addWidget(
            self.checkBoxTransitionProbability, 9, 3, 1, 1
        )
        self.examplesComboBox = QtWidgets.QComboBox(self.centralWidget)
        self.examplesComboBox.setMinimumSize(QtCore.QSize(100, 0))
        self.examplesComboBox.setMaximumSize(QtCore.QSize(100, 16777215))
        self.examplesComboBox.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.examplesComboBox.setFrame(True)
        self.examplesComboBox.setObjectName("examplesComboBox")
        self.examplesComboBox.addItem("")
        self.examplesComboBox.addItem("")
        self.examplesComboBox.addItem("")
        self.gridLayout.addWidget(self.examplesComboBox, 15, 1, 1, 1)
        self.inputMaxAggregateSize = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputMaxAggregateSize.sizePolicy().hasHeightForWidth()
        )
        self.inputMaxAggregateSize.setSizePolicy(sizePolicy)
        self.inputMaxAggregateSize.setMinimumSize(QtCore.QSize(0, 0))
        self.inputMaxAggregateSize.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputMaxAggregateSize.setFrame(True)
        self.inputMaxAggregateSize.setAlignment(
            QtCore.Qt.AlignLeading
            | QtCore.Qt.AlignLeft
            | QtCore.Qt.AlignVCenter
        )
        self.inputMaxAggregateSize.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputMaxAggregateSize.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputMaxAggregateSize.setKeyboardTracking(False)
        self.inputMaxAggregateSize.setMinimum(2)
        self.inputMaxAggregateSize.setMaximum(20)
        self.inputMaxAggregateSize.setProperty("value", 2)
        self.inputMaxAggregateSize.setObjectName("inputMaxAggregateSize")
        self.gridLayout.addWidget(self.inputMaxAggregateSize, 3, 1, 1, 1)
        self.checkBoxNoise = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxNoise.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxNoise.setObjectName("checkBoxNoise")
        self.gridLayout.addWidget(self.checkBoxNoise, 10, 3, 1, 1)
        self.inputNoiseHi = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputNoiseHi.sizePolicy().hasHeightForWidth()
        )
        self.inputNoiseHi.setSizePolicy(sizePolicy)
        self.inputNoiseHi.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputNoiseHi.setStyleSheet("")
        self.inputNoiseHi.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.inputNoiseHi.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputNoiseHi.setKeyboardTracking(False)
        self.inputNoiseHi.setProperty("value", 0.1)
        self.inputNoiseHi.setObjectName("inputNoiseHi")
        self.gridLayout.addWidget(self.inputNoiseHi, 10, 2, 1, 1)
        self.labelRandomStatesMax = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelRandomStatesMax.sizePolicy().hasHeightForWidth()
        )
        self.labelRandomStatesMax.setSizePolicy(sizePolicy)
        self.labelRandomStatesMax.setObjectName("labelRandomStatesMax")
        self.gridLayout.addWidget(self.labelRandomStatesMax, 5, 0, 1, 1)
        self.checkBoxRandomState = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxRandomState.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxRandomState.setObjectName("checkBoxRandomState")
        self.gridLayout.addWidget(self.checkBoxRandomState, 4, 3, 1, 1)
        self.labelMaxAggregateSize = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelMaxAggregateSize.sizePolicy().hasHeightForWidth()
        )
        self.labelMaxAggregateSize.setSizePolicy(sizePolicy)
        self.labelMaxAggregateSize.setObjectName("labelMaxAggregateSize")
        self.gridLayout.addWidget(self.labelMaxAggregateSize, 3, 0, 1, 1)
        self.inputScrambleProbability = QtWidgets.QDoubleSpinBox(
            self.centralWidget
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputScrambleProbability.sizePolicy().hasHeightForWidth()
        )
        self.inputScrambleProbability.setSizePolicy(sizePolicy)
        self.inputScrambleProbability.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputScrambleProbability.setStyleSheet("")
        self.inputScrambleProbability.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputScrambleProbability.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputScrambleProbability.setKeyboardTracking(False)
        self.inputScrambleProbability.setMaximum(1.0)
        self.inputScrambleProbability.setSingleStep(0.1)
        self.inputScrambleProbability.setProperty("value", 0.0)
        self.inputScrambleProbability.setObjectName("inputScrambleProbability")
        self.gridLayout.addWidget(self.inputScrambleProbability, 1, 1, 1, 1)
        self.labelScaler = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelScaler.sizePolicy().hasHeightForWidth()
        )
        self.labelScaler.setSizePolicy(sizePolicy)
        self.labelScaler.setObjectName("labelScaler")
        self.gridLayout.addWidget(self.labelScaler, 13, 0, 1, 1)
        self.inputTransitionProbabilityLo = QtWidgets.QDoubleSpinBox(
            self.centralWidget
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputTransitionProbabilityLo.sizePolicy().hasHeightForWidth()
        )
        self.inputTransitionProbabilityLo.setSizePolicy(sizePolicy)
        self.inputTransitionProbabilityLo.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputTransitionProbabilityLo.setStyleSheet("")
        self.inputTransitionProbabilityLo.setWrapping(False)
        self.inputTransitionProbabilityLo.setFrame(True)
        self.inputTransitionProbabilityLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputTransitionProbabilityLo.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputTransitionProbabilityLo.setKeyboardTracking(False)
        self.inputTransitionProbabilityLo.setMaximum(1.0)
        self.inputTransitionProbabilityLo.setSingleStep(0.1)
        self.inputTransitionProbabilityLo.setProperty("value", 0.05)
        self.inputTransitionProbabilityLo.setObjectName(
            "inputTransitionProbabilityLo"
        )
        self.gridLayout.addWidget(self.inputTransitionProbabilityLo, 9, 1, 1, 1)
        self.inputScalerLo = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputScalerLo.sizePolicy().hasHeightForWidth()
        )
        self.inputScalerLo.setSizePolicy(sizePolicy)
        self.inputScalerLo.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputScalerLo.setStyleSheet("")
        self.inputScalerLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputScalerLo.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputScalerLo.setKeyboardTracking(False)
        self.inputScalerLo.setMinimum(1)
        self.inputScalerLo.setMaximum(99999)
        self.inputScalerLo.setObjectName("inputScalerLo")
        self.gridLayout.addWidget(self.inputScalerLo, 13, 1, 1, 1)
        self.inputMaxRandomStates = QtWidgets.QSpinBox(self.centralWidget)
        self.inputMaxRandomStates.setEnabled(False)
        self.inputMaxRandomStates.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputMaxRandomStates.setFrame(False)
        self.inputMaxRandomStates.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputMaxRandomStates.setKeyboardTracking(True)
        self.inputMaxRandomStates.setMinimum(1)
        self.inputMaxRandomStates.setMaximum(5)
        self.inputMaxRandomStates.setProperty("value", 5)
        self.inputMaxRandomStates.setObjectName("inputMaxRandomStates")
        self.gridLayout.addWidget(self.inputMaxRandomStates, 5, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.gridLayout.addItem(spacerItem, 14, 0, 1, 1)
        self.inputTransitionProbabilityHi = QtWidgets.QDoubleSpinBox(
            self.centralWidget
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputTransitionProbabilityHi.sizePolicy().hasHeightForWidth()
        )
        self.inputTransitionProbabilityHi.setSizePolicy(sizePolicy)
        self.inputTransitionProbabilityHi.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputTransitionProbabilityHi.setStyleSheet("")
        self.inputTransitionProbabilityHi.setWrapping(False)
        self.inputTransitionProbabilityHi.setFrame(True)
        self.inputTransitionProbabilityHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputTransitionProbabilityHi.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputTransitionProbabilityHi.setKeyboardTracking(False)
        self.inputTransitionProbabilityHi.setMaximum(1.0)
        self.inputTransitionProbabilityHi.setSingleStep(0.1)
        self.inputTransitionProbabilityHi.setProperty("value", 0.1)
        self.inputTransitionProbabilityHi.setObjectName(
            "inputTransitionProbabilityHi"
        )
        self.gridLayout.addWidget(self.inputTransitionProbabilityHi, 9, 2, 1, 1)
        self.inputAggregateProbability = QtWidgets.QDoubleSpinBox(
            self.centralWidget
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputAggregateProbability.sizePolicy().hasHeightForWidth()
        )
        self.inputAggregateProbability.setSizePolicy(sizePolicy)
        self.inputAggregateProbability.setMaximumSize(
            QtCore.QSize(100, 16777215)
        )
        self.inputAggregateProbability.setStyleSheet("")
        self.inputAggregateProbability.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputAggregateProbability.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputAggregateProbability.setKeyboardTracking(False)
        self.inputAggregateProbability.setMaximum(1.0)
        self.inputAggregateProbability.setSingleStep(0.1)
        self.inputAggregateProbability.setProperty("value", 0.0)
        self.inputAggregateProbability.setObjectName(
            "inputAggregateProbability"
        )
        self.gridLayout.addWidget(self.inputAggregateProbability, 2, 1, 1, 1)
        self.labelAAmismatch_2 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelAAmismatch_2.sizePolicy().hasHeightForWidth()
        )
        self.labelAAmismatch_2.setSizePolicy(sizePolicy)
        self.labelAAmismatch_2.setObjectName("labelAAmismatch_2")
        self.gridLayout.addWidget(self.labelAAmismatch_2, 12, 0, 1, 1)
        self.labelBlinkingProbability = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelBlinkingProbability.sizePolicy().hasHeightForWidth()
        )
        self.labelBlinkingProbability.setSizePolicy(sizePolicy)
        self.labelBlinkingProbability.setObjectName("labelBlinkingProbability")
        self.gridLayout.addWidget(self.labelBlinkingProbability, 8, 0, 1, 1)
        self.labelNumberOfTraces_2 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelNumberOfTraces_2.sizePolicy().hasHeightForWidth()
        )
        self.labelNumberOfTraces_2.setSizePolicy(sizePolicy)
        self.labelNumberOfTraces_2.setText("")
        self.labelNumberOfTraces_2.setObjectName("labelNumberOfTraces_2")
        self.gridLayout.addWidget(self.labelNumberOfTraces_2, 17, 0, 1, 1)
        self.labelExport = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelExport.sizePolicy().hasHeightForWidth()
        )
        self.labelExport.setSizePolicy(sizePolicy)
        self.labelExport.setObjectName("labelExport")
        self.gridLayout.addWidget(self.labelExport, 16, 0, 1, 1)
        self.inputBleedthroughLo = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputBleedthroughLo.sizePolicy().hasHeightForWidth()
        )
        self.inputBleedthroughLo.setSizePolicy(sizePolicy)
        self.inputBleedthroughLo.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputBleedthroughLo.setStyleSheet("")
        self.inputBleedthroughLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputBleedthroughLo.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputBleedthroughLo.setKeyboardTracking(False)
        self.inputBleedthroughLo.setMinimum(0.0)
        self.inputBleedthroughLo.setMaximum(1.0)
        self.inputBleedthroughLo.setProperty("value", 0.0)
        self.inputBleedthroughLo.setObjectName("inputBleedthroughLo")
        self.gridLayout.addWidget(self.inputBleedthroughLo, 12, 1, 1, 1)
        self.checkBoxALifetime = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxALifetime.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxALifetime.setObjectName("checkBoxALifetime")
        self.gridLayout.addWidget(self.checkBoxALifetime, 7, 3, 1, 1)
        self.inputScalerHi = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputScalerHi.sizePolicy().hasHeightForWidth()
        )
        self.inputScalerHi.setSizePolicy(sizePolicy)
        self.inputScalerHi.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputScalerHi.setStyleSheet("")
        self.inputScalerHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputScalerHi.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputScalerHi.setKeyboardTracking(False)
        self.inputScalerHi.setMinimum(1)
        self.inputScalerHi.setMaximum(99999)
        self.inputScalerHi.setObjectName("inputScalerHi")
        self.gridLayout.addWidget(self.inputScalerHi, 13, 2, 1, 1)
        self.labelDisplay = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelDisplay.sizePolicy().hasHeightForWidth()
        )
        self.labelDisplay.setSizePolicy(sizePolicy)
        self.labelDisplay.setObjectName("labelDisplay")
        self.gridLayout.addWidget(self.labelDisplay, 15, 0, 1, 1)
        self.labelAcceptorMeanLifetime = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelAcceptorMeanLifetime.sizePolicy().hasHeightForWidth()
        )
        self.labelAcceptorMeanLifetime.setSizePolicy(sizePolicy)
        self.labelAcceptorMeanLifetime.setObjectName(
            "labelAcceptorMeanLifetime"
        )
        self.gridLayout.addWidget(self.labelAcceptorMeanLifetime, 7, 0, 1, 1)
        self.inputBleedthroughHi = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.inputBleedthroughHi.sizePolicy().hasHeightForWidth()
        )
        self.inputBleedthroughHi.setSizePolicy(sizePolicy)
        self.inputBleedthroughHi.setMaximumSize(QtCore.QSize(100, 16777215))
        self.inputBleedthroughHi.setStyleSheet("")
        self.inputBleedthroughHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.inputBleedthroughHi.setCorrectionMode(
            QtWidgets.QAbstractSpinBox.CorrectToPreviousValue
        )
        self.inputBleedthroughHi.setKeyboardTracking(False)
        self.inputBleedthroughHi.setMinimum(0.0)
        self.inputBleedthroughHi.setMaximum(1.0)
        self.inputBleedthroughHi.setProperty("value", 0.0)
        self.inputBleedthroughHi.setObjectName("inputBleedthroughHi")
        self.gridLayout.addWidget(self.inputBleedthroughHi, 12, 2, 1, 1)
        self.labelTraceLength = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelTraceLength.sizePolicy().hasHeightForWidth()
        )
        self.labelTraceLength.setSizePolicy(sizePolicy)
        self.labelTraceLength.setObjectName("labelTraceLength")
        self.gridLayout.addWidget(self.labelTraceLength, 0, 0, 1, 1)
        self.checkBoxMismatch = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBoxMismatch.setMaximumSize(QtCore.QSize(100, 16777215))
        self.checkBoxMismatch.setObjectName("checkBoxMismatch")
        self.gridLayout.addWidget(self.checkBoxMismatch, 11, 3, 1, 1)
        self.labelFretStateMeans = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelFretStateMeans.sizePolicy().hasHeightForWidth()
        )
        self.labelFretStateMeans.setSizePolicy(sizePolicy)
        self.labelFretStateMeans.setObjectName("labelFretStateMeans")
        self.gridLayout.addWidget(self.labelFretStateMeans, 4, 0, 1, 1)
        self.pushButtonExport = QtWidgets.QPushButton(self.centralWidget)
        self.pushButtonExport.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonExport.setObjectName("pushButtonExport")
        self.gridLayout.addWidget(self.pushButtonExport, 16, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 1, 1)
        SimulatorWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(SimulatorWindow)
        QtCore.QMetaObject.connectSlotsByName(SimulatorWindow)
        SimulatorWindow.setTabOrder(
            self.inputTraceLength, self.inputScrambleProbability
        )
        SimulatorWindow.setTabOrder(
            self.inputScrambleProbability, self.inputAggregateProbability
        )
        SimulatorWindow.setTabOrder(
            self.inputAggregateProbability, self.inputMaxAggregateSize
        )
        SimulatorWindow.setTabOrder(
            self.inputMaxAggregateSize, self.inputFretStateMeans
        )
        SimulatorWindow.setTabOrder(
            self.inputFretStateMeans, self.checkBoxRandomState
        )
        SimulatorWindow.setTabOrder(
            self.checkBoxRandomState, self.inputMaxRandomStates
        )
        SimulatorWindow.setTabOrder(
            self.inputMaxRandomStates, self.inputDonorMeanLifetime
        )
        SimulatorWindow.setTabOrder(
            self.inputDonorMeanLifetime, self.checkBoxDlifetime
        )
        SimulatorWindow.setTabOrder(
            self.checkBoxDlifetime, self.inputAcceptorMeanLifetime
        )
        SimulatorWindow.setTabOrder(
            self.inputAcceptorMeanLifetime, self.checkBoxALifetime
        )
        SimulatorWindow.setTabOrder(
            self.checkBoxALifetime, self.inputBlinkingProbability
        )
        SimulatorWindow.setTabOrder(
            self.inputBlinkingProbability, self.inputTransitionProbabilityLo
        )
        SimulatorWindow.setTabOrder(
            self.inputTransitionProbabilityLo, self.inputTransitionProbabilityHi
        )
        SimulatorWindow.setTabOrder(
            self.inputTransitionProbabilityHi,
            self.checkBoxTransitionProbability,
        )
        SimulatorWindow.setTabOrder(
            self.checkBoxTransitionProbability, self.inputNoiseLo
        )
        SimulatorWindow.setTabOrder(self.inputNoiseLo, self.inputNoiseHi)
        SimulatorWindow.setTabOrder(self.inputNoiseHi, self.checkBoxNoise)
        SimulatorWindow.setTabOrder(self.checkBoxNoise, self.inputMismatchLo)
        SimulatorWindow.setTabOrder(self.inputMismatchLo, self.inputMismatchHi)
        SimulatorWindow.setTabOrder(self.inputMismatchHi, self.checkBoxMismatch)
        SimulatorWindow.setTabOrder(
            self.checkBoxMismatch, self.inputBleedthroughLo
        )
        SimulatorWindow.setTabOrder(
            self.inputBleedthroughLo, self.inputBleedthroughHi
        )
        SimulatorWindow.setTabOrder(
            self.inputBleedthroughHi, self.checkBoxBleedthrough
        )
        SimulatorWindow.setTabOrder(
            self.checkBoxBleedthrough, self.inputScalerLo
        )
        SimulatorWindow.setTabOrder(self.inputScalerLo, self.inputScalerHi)
        SimulatorWindow.setTabOrder(self.inputScalerHi, self.checkBoxScaler)
        SimulatorWindow.setTabOrder(self.checkBoxScaler, self.examplesComboBox)
        SimulatorWindow.setTabOrder(
            self.examplesComboBox, self.inputNumberOfTraces
        )
        SimulatorWindow.setTabOrder(
            self.inputNumberOfTraces, self.pushButtonExport
        )

    def retranslateUi(self, SimulatorWindow):
        _translate = QtCore.QCoreApplication.translate
        self.labelScrambleProbability.setText(
            _translate("SimulatorWindow", "Scramble Probability")
        )
        self.checkBoxScaler.setText(
            _translate("SimulatorWindow", "Single value")
        )
        self.labelTransitionProbability.setText(
            _translate("SimulatorWindow", "Transition Probability")
        )
        self.labelAAmismatch.setText(
            _translate("SimulatorWindow", "Acceptor-only mismatch")
        )
        self.labelAggregateProbability.setText(
            _translate("SimulatorWindow", "Aggregate Probability")
        )
        self.inputFretStateMeans.setText(_translate("SimulatorWindow", "0.5"))
        self.checkBoxBleedthrough.setText(
            _translate("SimulatorWindow", "Single value")
        )
        self.labelDonorMeanLifetime.setText(
            _translate("SimulatorWindow", "Donor Mean Lifetime")
        )
        self.labelNoise.setText(_translate("SimulatorWindow", "Noise"))
        self.checkBoxDlifetime.setText(
            _translate("SimulatorWindow", "No bleach")
        )
        self.checkBoxTransitionProbability.setText(
            _translate("SimulatorWindow", "Single value")
        )
        self.examplesComboBox.setItemText(
            0, _translate("SimulatorWindow", "2x2")
        )
        self.examplesComboBox.setItemText(
            1, _translate("SimulatorWindow", "3x3")
        )
        self.examplesComboBox.setItemText(
            2, _translate("SimulatorWindow", "4x4")
        )
        self.checkBoxNoise.setText(
            _translate("SimulatorWindow", "Single value")
        )
        self.labelRandomStatesMax.setText(
            _translate("SimulatorWindow", "Max Number of Random States")
        )
        self.checkBoxRandomState.setText(
            _translate("SimulatorWindow", "Random")
        )
        self.labelMaxAggregateSize.setText(
            _translate("SimulatorWindow", "Max Aggregate Size")
        )
        self.labelScaler.setText(_translate("SimulatorWindow", "Post scaling"))
        self.labelAAmismatch_2.setText(
            _translate("SimulatorWindow", "Donor bleedthrough")
        )
        self.labelBlinkingProbability.setText(
            _translate("SimulatorWindow", "Blinking Probability")
        )
        self.labelExport.setText(_translate("SimulatorWindow", "Export Traces"))
        self.checkBoxALifetime.setText(
            _translate("SimulatorWindow", "No bleach")
        )
        self.labelDisplay.setText(
            _translate("SimulatorWindow", "Display Examples")
        )
        self.labelAcceptorMeanLifetime.setText(
            _translate("SimulatorWindow", "Acceptor Mean Lifetime")
        )
        self.labelTraceLength.setText(
            _translate("SimulatorWindow", "Trace Length")
        )
        self.checkBoxMismatch.setText(
            _translate("SimulatorWindow", "Single value")
        )
        self.labelFretStateMeans.setText(
            _translate("SimulatorWindow", "FRET State Means")
        )
        self.pushButtonExport.setText(_translate("SimulatorWindow", "Export"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    SimulatorWindow = QtWidgets.QMainWindow()
    ui = Ui_SimulatorWindow()
    ui.setupUi(SimulatorWindow)
    SimulatorWindow.show()
    sys.exit(app.exec_())
