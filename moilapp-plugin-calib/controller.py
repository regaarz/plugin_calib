import os
import cv2 as cv
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .ui_calib import Ui_Form
from src.plugin_interface import PluginInterface

class MainWindow(QWidget):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = model
        self.set_stylesheet()

        self.ui.calibration.clicked.connect(self.calibrate_camera)
        self.ui.camera.clicked.connect(self.detect_checker_board)
        self.ui.capture.clicked.connect(self.capture_camera)

        self.ui.spinBox_X.valueChanged.connect(self.update_dimension)
        self.ui.spinBox_Y.valueChanged.connect(self.update_dimension)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.image_dir_path = "images"
        if not os.path.isdir(self.image_dir_path):
            os.makedirs(self.image_dir_path)
            print(f'"{self.image_dir_path}" Directory is created')
        else:
            print(f'"{self.image_dir_path}" Directory already Exists.')

        self.CHESS_BOARD_DIM = (self.ui.spinBox_X.value(), self.ui.spinBox_Y.value())

        # Image counter
        self.image_counter = 0
        self.cap = None
        self.camera_started = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Setup matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout = QVBoxLayout(self.ui.label_3)
        self.plot_layout.addWidget(self.canvas)

    def set_stylesheet(self):
        self.ui.label.setStyleSheet("font-size:12pt;")
        self.ui.label.setStyleSheet(self.model.style_label())

    def update_dimension(self):
        x_value = self.ui.spinBox_X.value()
        y_value = self.ui.spinBox_Y.value()
        self.CHESS_BOARD_DIM = (x_value, y_value)

    def load_calibration_data(self):
        SQUARE_SIZE = 14  # millimeters
        obj_3D = np.zeros((self.CHESS_BOARD_DIM[0] * self.CHESS_BOARD_DIM[1], 3), np.float32)
        obj_3D[:, :2] = np.mgrid[0:self.CHESS_BOARD_DIM[0], 0:self.CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
        obj_3D *= SQUARE_SIZE

        obj_points_3D = []
        img_points_2D = []

        files = os.listdir(self.image_dir_path)
        for file in files:
            imagePath = os.path.join(self.image_dir_path, file)
            image = cv.imread(imagePath)
            grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(image, self.CHESS_BOARD_DIM, None)
            if ret:
                obj_points_3D.append(obj_3D)
                corners2 = cv.cornerSubPix(grayScale, corners, (11, 11), (-1, -1), self.criteria)
                img_points_2D.append(corners2)

        cv.destroyAllWindows()

        if not obj_points_3D:
            return None, None, None, None

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
        )

        calib_data_path = "../calib_data"
        if not os.path.isdir(calib_data_path):
            os.makedirs(calib_data_path)
        np.savez(
            f"{calib_data_path}/MultiMatrix",
            camMatrix=mtx,
            distCoef=dist,
            rVector=rvecs,
            tVector=tvecs,
        )

        return mtx, dist, rvecs, tvecs

    def calibrate_camera(self):
        camMatrix, distCof, rVector, tVector = self.load_calibration_data()

        if camMatrix is None:
            self.ui.status.setText("No images found for calibration.")
            self.ui.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        self.ui.label_camMatrix.setText(str(camMatrix))
        self.ui.label_camMatrix.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_distCof.setText(str(distCof))
        self.ui.label_distCof.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_rVector.setText(str(rVector))
        self.ui.label_tVector.setText(str(tVector))

        self.plot_calibration_results(rVector, tVector)

    def plot_calibration_results(self, rvecs, tvecs):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv.Rodrigues(rvec)
            ax.scatter(tvec[0][0], tvec[0][1], tvec[0][2], c='r', marker='o')
            ax.plot([tvec[0][0], tvec[0][0] + R[0, 0]], [tvec[0][1], tvec[0][1] + R[1, 0]], [tvec[0][2], tvec[0][2] + R[2, 0]], c='b')
            ax.plot([tvec[0][0], tvec[0][0] + R[0, 1]], [tvec[0][1], tvec[0][1] + R[1, 1]], [tvec[0][2], tvec[0][2] + R[2, 1]], c='g')
            ax.plot([tvec[0][0], tvec[0][0] + R[0, 2]], [tvec[0][1], tvec[0][1] + R[1, 2]], [tvec[0][2], tvec[0][2] + R[2, 2]], c='b')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Calibration Results')
        self.canvas.draw()

    def detect_checker_board(self):
        self.cap = cv.VideoCapture(2)
        if not self.cap.isOpened():
            print("Error: Failed to open camera.")
            return
        else:
            print("Camera opened successfully")
        self.timer.start(10)
        self.camera_started = True

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if self.CHESS_BOARD_DIM[0] > 2 and self.CHESS_BOARD_DIM[1] > 2:
                    ret, corners = cv.findChessboardCorners(gray, self.CHESS_BOARD_DIM)
                    if ret:
                        corners = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
                        frame = cv.drawChessboardCorners(frame, self.CHESS_BOARD_DIM, corners, ret)
                else:
                    print("Not yet Chessboard")
                frame_resized = cv.resize(frame, (self.ui.cam.width(), self.ui.cam.height()))
                h, w, ch = frame_resized.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(q_img)
                self.ui.cam.setPixmap(pixmap)
                self.ui.cam.setScaledContents(True)

    def capture_camera(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, _ = cv.findChessboardCorners(gray, self.CHESS_BOARD_DIM)
                if ret:
                    self.image_counter += 1
                    filename = f"{self.image_dir_path}/image_{self.image_counter}.jpg"
                    cv.imwrite(filename, frame)
                    print(f"Image saved: {filename}")
                    self.ui.status.setText("Image saved: " + filename)

    def close_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.ui.cam.clear()

    def __del__(self):
        self.close_camera()
