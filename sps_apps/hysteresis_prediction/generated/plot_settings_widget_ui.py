# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/opt/home/lua/code/sps-app-hysteresis/resources/ui/plot_settings_widget.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PlotSettingsWidget(object):
    def setupUi(self, PlotSettingsWidget):
        PlotSettingsWidget.setObjectName("PlotSettingsWidget")
        PlotSettingsWidget.resize(241, 325)
        PlotSettingsWidget.setMaximumSize(QtCore.QSize(300, 16777215))
        self.verticalLayout = QtWidgets.QVBoxLayout(PlotSettingsWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(PlotSettingsWidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QFormLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.labelTimespan = QtWidgets.QLabel(self.frame)
        self.labelTimespan.setObjectName("labelTimespan")
        self.formLayout.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.labelTimespan
        )
        self.spinBoxTimespan = QtWidgets.QSpinBox(self.frame)
        self.spinBoxTimespan.setMinimum(1)
        self.spinBoxTimespan.setMaximum(360)
        self.spinBoxTimespan.setSingleStep(10)
        self.spinBoxTimespan.setProperty("value", 60)
        self.spinBoxTimespan.setObjectName("spinBoxTimespan")
        self.formLayout.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.spinBoxTimespan
        )
        self.buttonResetAxis = QtWidgets.QPushButton(self.frame)
        self.buttonResetAxis.setObjectName("buttonResetAxis")
        self.formLayout.setWidget(
            2, QtWidgets.QFormLayout.SpanningRole, self.buttonResetAxis
        )
        self.labelDownsample = QtWidgets.QLabel(self.frame)
        self.labelDownsample.setObjectName("labelDownsample")
        self.formLayout.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.labelDownsample
        )
        self.spinBoxDownsample = QtWidgets.QSpinBox(self.frame)
        self.spinBoxDownsample.setMinimum(1)
        self.spinBoxDownsample.setMaximum(1000)
        self.spinBoxDownsample.setStepType(
            QtWidgets.QAbstractSpinBox.DefaultStepType
        )
        self.spinBoxDownsample.setObjectName("spinBoxDownsample")
        self.formLayout.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.spinBoxDownsample
        )
        self.verticalLayout.addWidget(self.frame)

        self.retranslateUi(PlotSettingsWidget)
        QtCore.QMetaObject.connectSlotsByName(PlotSettingsWidget)

    def retranslateUi(self, PlotSettingsWidget):
        _translate = QtCore.QCoreApplication.translate
        PlotSettingsWidget.setWindowTitle(
            _translate("PlotSettingsWidget", "Scrolling Plot Settings")
        )
        self.labelTimespan.setText(
            _translate("PlotSettingsWidget", "Time span [s]")
        )
        self.buttonResetAxis.setText(
            _translate("PlotSettingsWidget", "Reset Axis")
        )
        self.labelDownsample.setText(
            _translate("PlotSettingsWidget", "Downsample")
        )
