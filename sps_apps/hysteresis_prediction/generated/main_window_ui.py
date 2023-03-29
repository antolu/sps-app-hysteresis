# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/anton/code/sps-app-hysteresis/resources/ui/main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.resize(800, 600)
        main_window.setProperty("useRBAC", False)
        self.central_widget = QtWidgets.QWidget(main_window)
        self.central_widget.setObjectName("central_widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.central_widget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addWidget(self.frame)
        main_window.setCentralWidget(self.central_widget)
        self.menu_bar = QtWidgets.QMenuBar(main_window)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 800, 19))
        self.menu_bar.setObjectName("menu_bar")
        self.menu_file = QtWidgets.QMenu(self.menu_bar)
        self.menu_file.setObjectName("menu_file")
        self.menu_help = QtWidgets.QMenu(self.menu_bar)
        self.menu_help.setObjectName("menu_help")
        self.menu_view = QtWidgets.QMenu(self.menu_bar)
        self.menu_view.setObjectName("menu_view")
        main_window.setMenuBar(self.menu_bar)
        self.status_bar = QtWidgets.QStatusBar(main_window)
        self.status_bar.setObjectName("status_bar")
        main_window.setStatusBar(self.status_bar)
        self.action_exit = QtWidgets.QAction(main_window)
        self.action_exit.setMenuRole(QtWidgets.QAction.QuitRole)
        self.action_exit.setObjectName("action_exit")
        self.action_about = QtWidgets.QAction(main_window)
        self.action_about.setMenuRole(QtWidgets.QAction.AboutRole)
        self.action_about.setObjectName("action_about")
        self.action_minimize = QtWidgets.QAction(main_window)
        self.action_minimize.setObjectName("action_minimize")
        self.action_maximize = QtWidgets.QAction(main_window)
        self.action_maximize.setObjectName("action_maximize")
        self.action_fullscreen = QtWidgets.QAction(main_window)
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setObjectName("action_fullscreen")
        self.menu_file.addAction(self.action_exit)
        self.menu_help.addAction(self.action_about)
        self.menu_view.addAction(self.action_minimize)
        self.menu_view.addAction(self.action_maximize)
        self.menu_view.addAction(self.action_fullscreen)
        self.menu_view.addSeparator()
        self.menu_bar.addAction(self.menu_file.menuAction())
        self.menu_bar.addAction(self.menu_view.menuAction())
        self.menu_bar.addAction(self.menu_help.menuAction())

        self.retranslateUi(main_window)
        self.action_about.triggered.connect(main_window.showAboutDialog)
        self.action_exit.triggered.connect(main_window.close)
        self.action_minimize.triggered.connect(main_window.showMinimized)
        self.action_maximize.triggered.connect(main_window.showMaximized)
        self.action_fullscreen.triggered.connect(main_window.toggleFullScreen)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "SPS Hysteresis Prediction"))
        self.pushButton.setText(_translate("main_window", "PushButton"))
        self.menu_file.setTitle(_translate("main_window", "File"))
        self.menu_help.setTitle(_translate("main_window", "Help"))
        self.menu_view.setTitle(_translate("main_window", "View"))
        self.action_exit.setText(_translate("main_window", "Exit"))
        self.action_about.setText(_translate("main_window", "About"))
        self.action_minimize.setText(_translate("main_window", "Minimize"))
        self.action_maximize.setText(_translate("main_window", "Maximize"))
        self.action_fullscreen.setText(_translate("main_window", "Toggle Fullscreen"))
        self.action_fullscreen.setToolTip(_translate("main_window", "Toggle Fullscreen"))
from accwidgets.app_frame import ApplicationFrame
