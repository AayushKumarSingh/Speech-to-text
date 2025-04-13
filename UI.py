import sys
import warnings
import requests
import sounddevice as sd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.io.wavfile import write
from PyQt5 import QtCore, QtGui, QtWidgets


warnings.filterwarnings('ignore')


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        self.finishReady = False
        self.timer = QtCore.QTimer()
        self.recording = False
        self.audio_data = []
        self.samplerate = 16000

        Dialog.setObjectName("Dialog")
        Dialog.resize(776, 513)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        Dialog.setFont(font)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 771, 511))
        font.setPointSize(9)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")

        # Train Model Tab
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.finishRec = QtWidgets.QPushButton(self.tab)
        self.finishRec.setEnabled(False)
        self.finishRec.setGeometry(QtCore.QRect(590, 390, 121, 41))
        font.setPointSize(9)
        self.finishRec.setFont(font)
        self.finishRec.setObjectName("finishRec")

        self.startRec = QtWidgets.QPushButton(self.tab)
        self.startRec.setGeometry(QtCore.QRect(450, 390, 121, 41))
        self.startRec.setFont(font)
        self.startRec.setObjectName("startRec")

        self.progressBar = QtWidgets.QProgressBar(self.tab)
        self.progressBar.setEnabled(True)
        self.progressBar.setGeometry(QtCore.QRect(520, 350, 181, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.text = QtWidgets.QTextBrowser(self.tab)
        self.text.setGeometry(QtCore.QRect(20, 120, 721, 211))
        self.text.setObjectName("text")
        font.setPointSize(14)
        font.setBold(False)
        self.text.setFont(font)
        self.text.setText("In recent years, technological advancements have significantly influenced various aspects of human life. From artificial intelligence to biotechnology, innovation continues to reshape industries at an unprecedented pace. However, with progress comes challenges, such as ethical considerations, data privacy concerns, and the potential impact on employment. Addressing these issues requires a balanced approach that fosters both innovation and responsibility.")

        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(180, 20, 341, 61))
        font.setPointSize(13)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(200, 90, 311, 16))
        self.label_2.setObjectName("label_2")

        self.tabWidget.addTab(self.tab, "Train Model")

        # Predicted Output Tab
        self.tab_predicted = QtWidgets.QWidget()
        self.tab_predicted.setObjectName("tab_predicted")

        self.predictedLabel = QtWidgets.QLabel(self.tab_predicted)
        self.predictedLabel.setGeometry(QtCore.QRect(20, 20, 341, 31))
        font.setPointSize(14)
        self.predictedLabel.setFont(font)
        self.predictedLabel.setObjectName("predictedLabel")

        self.progressBar1 = QtWidgets.QProgressBar(self.tab_predicted)
        self.progressBar1.setEnabled(True)
        self.progressBar1.setGeometry(QtCore.QRect(310, 390, 250, 35))
        self.progressBar1.setProperty("value", 0)
        self.progressBar1.setObjectName("progressBar")
        self.progressBar1.hide()

        font.setPointSize(18)
        font.setBold(False)
        self.predictedText = QtWidgets.QTextBrowser(self.tab_predicted)
        self.predictedText.setGeometry(QtCore.QRect(20, 60, 721, 300))
        self.predictedText.setText("Start Prediction for transcript! ")
        self.predictedText.setObjectName("predictedText")
        self.predictedText.setFont(font)

        font.setPointSize(10)
        self.startPrediction = QtWidgets.QPushButton(self.tab_predicted)
        self.startPrediction.setGeometry(QtCore.QRect(310, 390, 150, 41))
        self.startPrediction.setFont(font)
        self.startPrediction.setObjectName("startPrediction")

        self.tabWidget.addTab(self.tab_predicted, "Predicted Output")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.progressBar.hide()
        self.progressBar.setEnabled(False)

        self.startRec.clicked.connect(self.onClickStartRec)
        self.finishRec.clicked.connect(self.onClickFinishRec)
        self.startPrediction.clicked.connect(self.onClickStartPrediction)
        self.timer.timeout.connect(self.updateProgress)
        self.timer.timeout.connect(self.updateProgressPred)


    def audio_callback(self, indata, frames, time, status):
        """ Callback function to store audio data """
        if status:
            print(f"Audio Status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())  # Store audio data

    def start_recording(self):
        """ Start recording """
        self.audio_data = []
        self.recording = True
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype=np.float32, callback=self.audio_callback)
        self.stream.start()
        print("Recording started")

    def stop_recording(self):
        """ Stop recording and save audio file """
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()

            if self.audio_data:
                audio_np = np.concatenate(self.audio_data, axis=0)
                audio_int16 = np.int16(audio_np * 32767)  # Convert to 16-bit PCM
                write("output.wav", self.samplerate, audio_int16)  # Save as WAV
                print("Recording saved (Check 'output.wav')")

    def onClickStartRec(self):
        self.finishReady = True
        self.finishRec.setEnabled(True)
        self.startRec.setEnabled(False)
        self.startRec.hide()
        self.progressBar.show()
        self.progressBar.setEnabled(True)
        self.progressBar.setValue(0)

        # Start audio recording
        QtCore.QTimer.singleShot(0, self.start_recording)

        self.timer.start(200)

    def updateProgress(self):
        if self.progressBar.value() < 100:
            self.progressBar.setValue(self.progressBar.value() + 1)
        else:
            self.timer.stop()
            self.onClickFinishRec()

    def updateProgressPred(self):
        if self.progressBar1.value() < 100:
            self.progressBar1.setValue(self.progressBar1.value() + 1)
        else:
            self.timer.stop()
            self.onClickFinishRec()


    def onClickFinishRec(self):
        if self.finishReady:
            self.timer.stop()
            self.progressBar.setValue(0)
            self.progressBar.hide()
            self.progressBar1.setValue(0)
            self.progressBar1.hide()
            self.startRec.setEnabled(True)
            self.startRec.show()
            self.finishRec.setEnabled(False)

            # Stop recording and save file
            self.stop_recording()

            self.predictedText.setText("Prediction started")

            # Start BackGround Task
            self.prediction_thread = PredictionThread()
            self.prediction_thread.result_signal.connect(self.updatePredictionResult)
            self.prediction_thread.start()

    def onClickStartPrediction(self):
        self.finishReady = True
        self.finishRec.setEnabled(True)
        self.startPrediction.setEnabled(False)
        self.startPrediction.hide()
        self.progressBar1.show()
        self.progressBar1.setEnabled(True)
        self.progressBar1.setValue(0)

        # Start audio recording
        QtCore.QTimer.singleShot(0, self.start_recording)

        self.timer.start(200)


    def updatePredictionResult(self, result):
        self.predictedText.setText(result)
        self.startPrediction.show()
        self.startPrediction.setEnabled(True)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Voice Recorder"))
        self.finishRec.setText(_translate("Dialog", "Finish"))
        self.startRec.setText(_translate("Dialog", "Start Recording"))
        self.startPrediction.setText(_translate("Dialog", "Start Prediction"))
        self.label.setText(_translate("Dialog", "Start Recording to train model on voice"))
        self.label_2.setText(_translate("Dialog", "Read the following Paragraph aloud into the microphone"))
        self.text.setText(_translate("Dialog", "In recent years, technological advancements have significantly influenced various aspects of human life. From artificial intelligence to biotechnology, innovation continues to reshape industries at an unprecedented pace. However, with progress comes challenges, such as ethical considerations, data privacy concerns, and the potential impact on employment. Addressing these issues requires a balanced approach that fosters both innovation and responsibility."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Train Model"))
        self.predictedLabel.setText(_translate("Dialog", "Predicted Transcript:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_predicted), _translate("Dialog", "Predicted Output"))


class PredictionThread(QThread):
    result_signal = pyqtSignal(str)

    def run(self):
        print("Prediction request sent")
        url = "http://127.0.0.1:5000/predict"
        files = {"file": open("output.wav", "rb")}

        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json().get("transcript", "No transcript available")
            else:
                result = f"Error: {response.status_code}"
        except Exception as e:
            result = f"API Error: {str(e)}"

        self.result_signal.emit(result)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())