# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rtsp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(667, 48)
        self.rtspEdit = QtWidgets.QLineEdit(Form)
        self.rtspEdit.setGeometry(QtCore.QRect(0, 10, 571, 20))
        self.rtspEdit.setObjectName("rtspEdit")
        self.okkButton = QtWidgets.QPushButton(Form)
        self.okkButton.setGeometry(QtCore.QRect(580, 10, 75, 23))
        self.okkButton.setObjectName("okkButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "输入rtsp地址"))
        self.okkButton.setText(_translate("Form", "确认"))

