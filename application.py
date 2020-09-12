#!/usr/bin/python3

import sys;
import cv2;
from PyQt5.QtCore import QTimer;
from PyQt5.QtGui import QImage, QPixmap;
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QWidget, QLabel, QFormLayout, QPushButton, QMessageBox;

class VideoDisplayWidget(QWidget):

  def __init__(self, parent, showControl = True):
  
    super(VideoDisplayWidget, self).__init__();
    self.cap = None;
    self.layout = QFormLayout(self)
    self.video_frame = QLabel();
    self.layout.addWidget(self.video_frame);
    if showControl:
      self.startButton = QPushButton('Start', parent);
      self.startButton.clicked.connect(self.start);
      self.startButton.setFixedWidth(50);
      self.pauseButton = QPushButton('Pause', parent);
      self.pauseButton.clicked.connect(self.pause);
      self.pauseButton.setFixedWidth(50);
      self.layout.addRow(self.startButton, self.pauseButton);
    self.setLayout(self.layout);

  def openVideo(self, filename):

    self.cap = cv2.VideoCapture(str(filename));

  def start(self):

    if self.cap is None:
      error = QMessageBox(self);
      error.setIcon(QMessageBox.Critical);
      error.setText("no video file opened!");
      error.setWindowTitle("Error");
      error.show();
      return;
    self.timer = QTimer();
    self.timer.timeout.connect(self.nextFrameHandler);
    self.timer.start(1000.0 / self.cap.get(cv2.CAP_PROP_FPS));

  def pause(self):

    self.timer.stop();

  def nextFrameHandler(self):

    ret, frame = self.cap.read();
    if ret == False:
      self.timer.stop();
      return;
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
    img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888);
    pix = QPixmap.fromImage(img);
    self.video_frame.setPixmap(pix);

class PlantarPressureWindow(QMainWindow):

  def __init__(self):
    
    super(PlantarPressureWindow, self).__init__();
    self.setWindowTitle('Planatar Pressure');
    self.setGeometry(10, 10, 640, 480);
    # setup menu
    self.mainMenu = self.menuBar();
    self.fileMenu = self.mainMenu.addMenu('&File');
    self.connectDevice = QAction("&Connect to Device", self);
    self.connectDevice.setShortcut("Ctrl+C");
    self.connectDevice.setStatusTip('Connect to Device');
    self.connectDevice.triggered.connect(self.connectDeviceHandler);
    self.disconnectDevice = QAction("&Disconnect", self);
    self.disconnectDevice.setShortcut("Ctrl+D");
    self.disconnectDevice.setStatusTip("Disconnect");
    self.disconnectDevice.triggered.connect(self.disconnectDeviceHandler);
    self.quit = QAction("&Quit", self);
    self.quit.setShortcut("Ctrl+Q");
    self.quit.setStatusTip("Quit");
    self.quit.triggered.connect(self.quitHandler);
    self.fileMenu.addAction(self.connectDevice);
    self.fileMenu.addAction(self.disconnectDevice);
    self.fileMenu.addAction(self.quit);
    self.pressureMenu = self.mainMenu.addMenu('&Pressure');
    self.showStaticPressure = QAction("&Static Pressure", self);
    self.showStaticPressure.setShortcut("Ctrl+S");
    self.showStaticPressure.setStatusTip("Show Static Pressure");
    self.showStaticPressure.triggered.connect(self.showStaticPressureHandler);
    self.showOnGoingPressure = QAction("&On Going Pressure", self);
    self.showOnGoingPressure.setShortcut("Ctrl+O");
    self.showOnGoingPressure.setStatusTip("Show On Going Pressure");
    self.showOnGoingPressure.triggered.connect(self.showOnGoingPressureHandler);
    self.pressureMenu.addAction(self.showStaticPressure);
    self.pressureMenu.addAction(self.showOnGoingPressure);
    # set video video widget
    self.videoDisplayWidget = VideoDisplayWidget(self);
    self.setCentralWidget(self.videoDisplayWidget);
    # set enable status
    self.setEnableStatus(False);
    
  def setEnableStatus(self, linked = True):

    if linked:
      self.connectDevice.setEnabled(False);
      self.disconnectDevice.setEnabled(True);
      self.showStaticPressure.setEnabled(True);
      self.showOnGoingPressure.setEnabled(True);
      self.videoDisplayWidget.setEnabled(True);
    else:
      self.connectDevice.setEnabled(True);
      self.disconnectDevice.setEnabled(False);
      self.showStaticPressure.setEnabled(False);
      self.showOnGoingPressure.setEnabled(False);
      self.videoDisplayWidget.setEnabled(False);
  
  def connectDeviceHandler(self):

    self.setEnableStatus(True);

  def disconnectDeviceHandler(self):

    self.setEnableStatus(False);

  def quitHandler(self):

    sys.exit(0);

  def showStaticPressureHandler(self):

    pass;

  def showOnGoingPressureHandler(self):

    pass;

if __name__ == "__main__":

  app = QApplication(sys.argv);
  window = PlantarPressureWindow();
  window.show();
  sys.exit(app.exec_());
