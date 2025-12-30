"""摄像头管理模块"""
import cv2
import numpy as np


class CameraManager:
    """摄像头管理类"""
    def __init__(self):
        self.current_camera = 0
        self.cap = None
        self.camera_properties = {
            'resolution_width': 320,  # 修改默认分辨率为320
            'resolution_height': 240,  # 修改默认分辨率为240
            'fps': 30,
            'exposure_mode': 'auto',  # 'auto' or 'manual'
            'exposure_value': -6,
            'calibration_matrix': None,
            'distortion_coeffs': None
        }

    def get_camera_list(self):
        """获取可用摄像头列表"""
        cameras = []
        for i in range(10):  # 检查前10个摄像头
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(f"Camera {i}")
                cap.release()
            else:
                break
        return cameras if cameras else ["No cameras found"]

    def start_camera(self, camera_index):
        """启动摄像头"""
        if self.cap is not None:
            self.cap.release()
        
        self.current_camera = int(camera_index.split(" ")[1]) if " " in camera_index else 0
        self.cap = cv2.VideoCapture(self.current_camera)
        
        if self.cap.isOpened():
            # 设置摄像头参数 - 先设置分辨率，再设置其他参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_properties['resolution_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_properties['resolution_height'])
            # 确保分辨率设置成功
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"设置分辨率: {self.camera_properties['resolution_width']}x{self.camera_properties['resolution_height']}")
            print(f"实际分辨率: {int(actual_width)}x{int(actual_height)}")
            
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_properties['fps'])
            
            # 设置曝光模式
            if self.camera_properties['exposure_mode'] == 'auto':
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_properties['exposure_value'])
            
            return f"Camera {self.current_camera} started successfully with resolution {int(actual_width)}x{int(actual_height)}"
        else:
            return "Failed to open camera"

    def update_camera_params(self, resolution_width, resolution_height, fps, exposure_mode, exposure_value):
        """更新摄像头参数"""
        self.camera_properties['resolution_width'] = int(resolution_width)
        self.camera_properties['resolution_height'] = int(resolution_height)
        self.camera_properties['fps'] = int(fps)
        self.camera_properties['exposure_mode'] = exposure_mode
        self.camera_properties['exposure_value'] = float(exposure_value)
        
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_properties['resolution_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_properties['resolution_height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_properties['fps'])
            
            if exposure_mode == 'auto':
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_properties['exposure_value'])
        
        return "Parameters updated"

    def update_distortion_params(self, k1, k2, p1, p2, k3):
        """更新畸变矫正参数"""
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.k3 = float(k3)
        return f"Distortion parameters updated: k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3}"

    def get_camera_frame(self, board_manager, calibration_data=None):
        """获取摄像头帧"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 检测标定板角点
                corners, ids = board_manager.detect_board_corners(gray)
                if board_manager.board_type == "chessboard" and corners is not None:
                    # 绘制棋盘格角点
                    frame = cv2.drawChessboardCorners(frame, board_manager.chessboard_size, corners, True)
                elif board_manager.board_type == "charuco" and corners is not None and ids is not None:
                    # 绘制 ChArUco 角点
                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
                
                # 如果已标定且存在标定矩阵和畸变系数，则应用畸变矫正
                if calibration_data and calibration_data.get('calibration_matrix') is not None and calibration_data.get('distortion_coeffs') is not None:
                    K = calibration_data['calibration_matrix']
                    D = calibration_data['distortion_coeffs']
                    
                    try:
                        # 确保畸变系数维度正确
                        if D.shape != (4, 1) and D.shape != (5, 1):  # 针孔模型通常是(5,1)，鱼眼模型是(4,1)
                            D = D.reshape(-1, 1)

                        # 判断是针孔模型还是鱼眼模型（根据畸变系数数量）
                        if D.shape[0] == 4:  # 鱼眼模型
                            # 鱼眼模型矫正
                            undistorted_frame = cv2.fisheye.undistortImage(frame, K, D, Knew=K)
                        else:  # 针孔模型 (D.shape[0] == 5 或其他)
                            # 针孔模型矫正
                            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (frame.shape[1], frame.shape[0]), 1, (frame.shape[1], frame.shape[0]))
                            undistorted_frame = cv2.undistort(frame, K, D, None, new_camera_matrix)

                        # 返回矫正后的帧
                        frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        print(f"[DEBUG] 畸变矫正失败: {e}")
                        # 如果矫正失败，返回原始帧
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # 如果没有标定数据，返回原始帧
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                return frame_rgb
            else:
                # 如果无法读取帧，返回适当大小的黑色图像
                return np.zeros((self.camera_properties['resolution_height'], self.camera_properties['resolution_width'], 3), dtype=np.uint8)
        else:
            # 如果摄像头未打开，返回适当大小的黑色图像
            return np.zeros((self.camera_properties['resolution_height'], self.camera_properties['resolution_width'], 3), dtype=np.uint8)

    def close_camera(self):
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None