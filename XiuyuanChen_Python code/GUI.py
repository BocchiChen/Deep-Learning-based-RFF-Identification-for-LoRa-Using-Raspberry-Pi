# -*- coding: utf-8 -*-
# pyuic5 -o GUI_SDR.py gui.ui

# GUI design
import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPalette, QBrush
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import receiver
import inference
from functools import partial
import GUI_SDR
import GUI_Wait
import GUI_CNN
import numpy as np
import matplotlib.pyplot as plt
   
def convert(ui):
    input1 = ui.spinBox.text()
    input2 = ui.comboBox_2.currentText()
    input3 = ui.lineEdit.text()   
    input4 = ui.lineEdit_2.text()
    input5 = ui.comboBox_3.currentText()
    input6 = ui.lineEdit_3.text()
    
    # Check whether input parameters have errors
    if int(input1) == 0:
        QMessageBox.warning(None,"Warning","The sampling time should not be zero.",QMessageBox.Yes)
    elif input3 == '':
        QMessageBox.warning(None,"Warning","The sampling frequency should not be empty.",QMessageBox.Yes)
    elif input3.replace(".", "").isdigit() != True:
        QMessageBox.warning(None,"Warning","The sampling frequency should be number.",QMessageBox.Yes)
    elif input4.replace(".", "").isdigit() != True:
        QMessageBox.warning(None,"Warning","The sampling length should be number.",QMessageBox.Yes)
    elif input4 == '':
        QMessageBox.warning(None,"Warning","The sampling length should not be empty.",QMessageBox.Yes)
    elif int(input4) < 10000:
        QMessageBox.warning(None,"Warning","The sampling length should be greater than 10000.",QMessageBox.Yes)
    elif input5.isdigit():
        input5 = float(input5)
    else:
        receive(int(input1),float(input3),int(input4),input5,float(input6))

def receive(s_tim,s_rate,length,gain,c_rate):
    # Extract and save LoRa packets
    input2 = ui.comboBox_2.currentText()
    if input2 != 'LoRa':
        QMessageBox.warning(None,"Warning","The application can only support LoRa network presently.",QMessageBox.Yes) 
    else:
        
        # Continue to the waiting stage
        ui2.lineEdit.setText(str(s_tim*0.8))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        ui2.lineEdit.setFont(font)
        ui2.lineEdit.setEnabled(False) 
        child.setPalette(window_pale)
        child.show()
         
        reply4 = QMessageBox.information(None,"Alert","Recheck the settings if required. The SDR will take a few seconds to collect packets. Press Yes to continue.",QMessageBox.Yes)   
        if reply4 == QMessageBox.Yes:
            # Receive packets through SDR        
            try:
                receiver.receivePackets(s_tim,s_rate,length,gain,c_rate) 
            except:
                # If some exceptions have occurred, popping up the warning window
                QMessageBox.warning(None,"Warning","No specified SDR is found.",QMessageBox.Yes) 
                MainWindow.close()
                child.close()
                app.quit()
                                   
            reply = QMessageBox.information(None,"Information","All packets haved been collected and processed. You can press Yes to check the spectrogram of each packet and do inference or press close to quit the system.",QMessageBox.Yes | QMessageBox.Close)   
            if reply == QMessageBox.Close:
                # If user chooses to quit the system
                reply2 = QMessageBox.question(None,"Really","You sure wanna quit the system?",QMessageBox.Yes | QMessageBox.No)  
                if reply2 == QMessageBox.Yes:
                    MainWindow.close()
                    child.close()
                    app.quit()
                else:
                    child.close()
                    ui3.lineEdit.setEnabled(False)
                    ui3.lineEdit_2.setEnabled(False)
                    ui3.lineEdit_3.setEnabled(False)
                    output.setPalette(window_pale)
                    readSpectrogram(0)
                    output.show()
                    
            elif reply == QMessageBox.Yes:
                # Continue to the inference stage
                child.close()
                ui3.lineEdit.setEnabled(False)
                ui3.lineEdit_2.setEnabled(False)
                ui3.lineEdit_3.setEnabled(False)
                output.setPalette(window_pale)
                readSpectrogram(0)
                output.show()
        else:
            MainWindow.close()
            child.close()
            app.quit()
     
def searchCNN():
    # Open the file explorer for users to search the NN model
    directory = QtWidgets.QFileDialog.getOpenFileName(None,"Open CNN Model","./","TFLite Files (*.tflite)")  
    ui3.lineEdit_2.setText(directory[0])
        
def readSpectrogram(num):
    # Read number of spectrograms
    loadData = np.load(r'./npydata/test/test1.npy')
    ui3.lineEdit_3.setText(str(loadData.shape[0]))
    ui3.spinBox.setMaximum(loadData.shape[0]-1)
    ui3.showSpect(num)
        
def inferLabel():
    # Infer the label of the transmitter
    if ui3.lineEdit_2.text() == '':
        QMessageBox.warning(None,"Warning","You must select a CNN model before using inference.",QMessageBox.Yes) 
    else:
        result = inference.infer(ui3.lineEdit_2.text())
        ui3.lineEdit.setText(str(result))

def modeChange():
    # Change the mode of wireless network if like
    mode = ui.comboBox_2.currentText()
    if mode == 'LoRa':
        ui.lineEdit_3.setText('868') 
    if mode == 'Bluetooth':
        ui.lineEdit_3.setText('2400')
    if mode == 'WiFi':
        ui.lineEdit_3.setText('5000')

# Main function
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    MainWindow = QMainWindow()
    ui = GUI_SDR.Ui_MainWindow()
    ui.setupUi(MainWindow)
    window_pale = QtGui.QPalette() 
    window_pale.setBrush(QPalette.Background, QtGui.QBrush(QtGui.QPixmap("C:/Users/13620/Desktop/pic3.jpg"))) 
    MainWindow.setPalette(window_pale)
    MainWindow.show()
    
    child = QDialog()
    ui2 = GUI_Wait.Ui_Dialog()
    ui2.setupUi(child)
    
    output = QWidget()
    ui3 = GUI_CNN.Ui_Widget()
    ui3.setupUi(output)
    
    QMessageBox.information(None,"Application User Guildance","Welcom to the RFFI system for LoRa. Here you can receive LoRa packets through RTL-SDR and infer the label of transmitting device using the CNN model.    Press Yes to continue.",QMessageBox.Yes)   
    modeChange()
    ui.comboBox_2.currentTextChanged.connect(modeChange)
    ui.pushButton.clicked.connect(partial(convert, ui))
    ui3.spinBox.valueChanged.connect(readSpectrogram,int(ui3.spinBox.value()))
    ui3.pushButton_2.clicked.connect(inferLabel)
    ui3.pushButton.clicked.connect(searchCNN)
    
    sys.exit(app.exec_())

