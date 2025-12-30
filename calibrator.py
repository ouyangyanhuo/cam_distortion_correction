import cv2
import numpy as np
import json
from pathlib import Path
import glob
import os


class CameraCalibrator:
    def __init__(self):
        # Charuco标定板参数
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.board = cv2.aruco.CharucoBoard((7, 10), 25.0, 17.5, self.dictionary)
        
        # 摄像头参数
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
        
        # 矫正参数
        self.k1 = 0.0  # 径向畸变系数
        self.k2 = 0.0  # 径向畸变系数
        self.p1 = 0.0  # 切向畸变系数
        self.p2 = 0.0  # 切向畸变系数
        self.k3 = 0.0  # 径向畸变系数
        
        # 标定相关
        self.calibration_images = []
        self.all_corners = []
        self.all_ids = []
        self.objpoints = []  # 3D点
        self.imgpoints = []  # 2D点
        self.board_type = "charuco"  # 标定板类型: "charuco" 或 "chessboard"
        self.chessboard_size = (9, 6)  # 棋盘格内角点数量 (cols, rows)
        self.square_size = 25.0  # 方格大小 (mm)
        
    def ensure_dir(self, p):
        """确保目录存在"""
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
    
    def save_calibration(self, out_path, model="pinhole", extra=None):
        """保存标定结果到YAML文件"""
        print(f"[DEBUG] 开始保存标定结果到: {out_path}")
        print(f"[DEBUG] 模型类型: {model}")
        
        if self.camera_properties['calibration_matrix'] is None or self.camera_properties['distortion_coeffs'] is None:
            print(f"[DEBUG] 没有标定数据可保存")
            return "没有标定数据可保存"
        
        print(f"[DEBUG] 标定矩阵信息:")
        print(f"[DEBUG]   - 内参矩阵K:\n{self.camera_properties['calibration_matrix']}")
        print(f"[DEBUG]   - 畸变系数D:\n{self.camera_properties['distortion_coeffs'].flatten()}")
        print(f"[DEBUG]   - 图像尺寸: {self.camera_properties['resolution_width']}x{self.camera_properties['resolution_height']}")
        
        fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", int(self.camera_properties['resolution_width']))
        fs.write("image_height", int(self.camera_properties['resolution_height']))
        fs.write("model", model)
        fs.write("K", self.camera_properties['calibration_matrix'])
        fs.write("D", self.camera_properties['distortion_coeffs'])
        if extra:
            print(f"[DEBUG] 保存额外信息:")
            for k, v in extra.items():
                print(f"[DEBUG]   - {k}: {v}")
                if isinstance(v, (int, float, str)):
                    fs.write(k, v)
                else:
                    fs.write(k, np.array(v))
        fs.release()
        print(f"[DEBUG] 标定结果已保存到: {out_path}")
        return f"标定结果已保存到: {out_path}"
        
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
    
    def detect_chessboard(self, gray, pattern_size):
        """检测棋盘格角点"""
        print(f"[DEBUG] 开始检测棋盘格角点，模式尺寸: {pattern_size}")
        corners = None
        ok = False
        if hasattr(cv2, "findChessboardCornersSB"):
            print(f"[DEBUG] 使用 findChessboardCornersSB 方法")
            ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        else:
            print(f"[DEBUG] 使用 findChessboardCorners 方法")
            ok, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        print(f"[DEBUG] 初始检测结果: ok={ok}, corners is not None: {corners is not None}")
        if not ok:
            print(f"[DEBUG] 棋盘格检测失败")
            return None
        print(f"[DEBUG] 检测到 {len(corners)} 个初始角点")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        print(f"[DEBUG] 亚像素优化后角点数量: {len(corners) if corners is not None else 0}")
        return corners
    
    def detect_charuco(self, gray):
        """
        兼容新旧 OpenCV 检测 ChArUco 板:
        - OpenCV 4.11+：CharucoDetector.detectBoard()（interpolateCornersCharuco 已移除）
        - 旧版 OpenCV：detectMarkers + interpolateCornersCharuco
        """
        # print(f"[DEBUG] 开始检测 ChArUco 角点")
        # 新 API
        if hasattr(cv2.aruco, "CharucoDetector"):
            # print(f"[DEBUG] 使用新 API: CharucoDetector")
            detector = cv2.aruco.CharucoDetector(self.board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            # print(f"[DEBUG] 新 API 检测结果: charuco_corners={charuco_corners is not None}, charuco_ids={charuco_ids is not None}")
            # if charuco_ids is not None:
            #     print(f"[DEBUG] 检测到 {len(charuco_ids)} 个 ChArUco ID")
            if charuco_ids is None or len(charuco_ids) < 6:
                # print(f"[DEBUG] ChArUco 检测失败: ID 数量不足")
                return None, None
            return charuco_corners, charuco_ids

        # 旧 API
        print(f"[DEBUG] 使用旧 API: detectMarkers + interpolateCornersCharuco")
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        
        print(f"[DEBUG] ArUco 标记检测结果: corners={corners is not None}, ids={ids is not None}")
        # if ids is not None:
        #     print(f"[DEBUG] 检测到 {len(ids)} 个 ArUco ID")

        if ids is None or len(ids) < 4:
            print(f"[DEBUG] ArUco 标记数量不足，无法进行 ChArUco 插值")
            return None, None

        if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
            # 极少数中间版本可能两者都没有
            print(f"[DEBUG] OpenCV 版本不支持 interpolateCornersCharuco")
            raise RuntimeError("当前 OpenCV aruco API 不完整：缺少 CharucoDetector 和 interpolateCornersCharuco。")

        ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=self.board
        )
        print(f"[DEBUG] interpolateCornersCharuco 结果: ok={ok}, charuco_corners={charuco_corners is not None}, charuco_ids={charuco_ids is not None}")
        if charuco_ids is not None:
            print(f"[DEBUG] 插值后 ChArUco ID 数量: {len(charuco_ids)}")
        
        if (not ok) or (charuco_ids is None) or (len(charuco_ids) < 6):
            print(f"[DEBUG] ChArUco 插值失败或ID数量不足")
            return None, None
        return charuco_corners, charuco_ids
    
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
    
    def get_camera_frame(self):
        """获取摄像头帧"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 检测标定板角点
                if self.board_type == "chessboard":
                    # 检测棋盘格角点
                    corners = self.detect_chessboard(gray, self.chessboard_size)
                    if corners is not None:
                        # 绘制棋盘格角点
                        frame = cv2.drawChessboardCorners(frame, self.chessboard_size, corners, True)
                else:  # charuco
                    # 检测 ChArUco 角点
                    charuco_corners, charuco_ids = self.detect_charuco(gray)
                    if charuco_corners is not None and charuco_ids is not None:
                        # 绘制 ChArUco 角点
                        frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                
                # 如果已标定且存在标定矩阵和畸变系数，则应用畸变矫正
                if (self.camera_properties['calibration_matrix'] is not None and 
                    self.camera_properties['distortion_coeffs'] is not None):
                    
                    # 应用畸变矫正
                    K = self.camera_properties['calibration_matrix']
                    D = self.camera_properties['distortion_coeffs']
                    
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
    
    def capture_calibration_image(self):
        """捕获用于标定的图像"""
        print(f"[DEBUG] 开始捕获标定图像，当前标定板类型: {self.board_type}")
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"[DEBUG] 成功读取图像，图像尺寸: {frame.shape}")
                
                if self.board_type == "chessboard":
                    print(f"[DEBUG] 检测棋盘格角点，尺寸: {self.chessboard_size}")
                    # 检测棋盘格角点
                    corners = self.detect_chessboard(gray, self.chessboard_size)
                    if corners is not None and len(corners) >= 4:
                        print(f"[DEBUG] 检测到 {len(corners)} 个棋盘格角点")
                        # 为棋盘格创建对象点
                        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
                        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
                        objp *= float(self.square_size)
                        print(f"[DEBUG] 创建了 {len(objp)} 个3D对象点")
                        
                        # 添加到标定数据
                        self.objpoints.append(objp)
                        self.imgpoints.append(corners)
                        self.calibration_images.append(gray)
                        print(f"[DEBUG] 已添加标定数据，当前共有 {len(self.calibration_images)} 张图像")
                        return f"Captured chessboard calibration image. Total: {len(self.calibration_images)}"
                    else:
                        print(f"[DEBUG] 未能检测到足够的棋盘格角点，检测到: {len(corners) if corners is not None else 0} 个")
                        return "Could not detect enough chessboard corners. Please adjust the board position."
                else:  # charuco
                    print(f"[DEBUG] 检测 ChArUco 角点")
                    # 检测 ChArUco 角点
                    charuco_corners, charuco_ids = self.detect_charuco(gray)
                    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 6:
                        print(f"[DEBUG] 检测到 {len(charuco_corners)} 个 ChArUco 角点, {len(charuco_ids)} 个 ID")
                        # 获取3D棋盘格角点
                        chess_corners_3d = self.board.getChessboardCorners()
                        
                        # 根据检测到的ID获取对应的3D坐标
                        ids = charuco_ids.flatten().astype(int)
                        obj = chess_corners_3d[ids, :]  # (M,3)
                        obj = obj.reshape(-1, 1, 3).astype(np.float64)
                        imgp = charuco_corners.reshape(-1, 1, 2).astype(np.float64)
                        
                        # 添加到标定数据
                        self.objpoints.append(obj)
                        self.imgpoints.append(imgp)
                        self.calibration_images.append(gray)
                        print(f"[DEBUG] 已添加标定数据，当前共有 {len(self.calibration_images)} 张图像")
                        return f"Captured charuco calibration image. Total: {len(self.calibration_images)}"
                    else:
                        print(f"[DEBUG] 未能检测到足够的 ChArUco 点，检测到角点: {len(charuco_corners) if charuco_corners is not None else 0}, ID: {len(charuco_ids) if charuco_ids is not None else 0}")
                        return "Could not detect enough charuco points. Please adjust the board position."
            else:
                print(f"[DEBUG] 无法读取摄像头帧")
                return "Failed to capture image"
        else:
            print(f"[DEBUG] 摄像头未打开")
            return "Camera is not opened"
    
    def calibrate_camera(self, model="pinhole"):
        """执行摄像头标定
        
        Args:
            model (str): 相机模型, 'pinhole' 或 'fisheye'
        """
        print(f"[DEBUG] 开始{model}模型标定过程")
        print(f"[DEBUG] 当前已捕获 {len(self.calibration_images)} 张标定图像")
        print(f"[DEBUG] 当前已收集 {len(self.objpoints)} 组3D-2D点对")
        
        if len(self.calibration_images) < 3:
            error_msg = f"标定失败: 图像数量不足。当前有 {len(self.calibration_images)} 张图像，需要至少3张。"
            print(error_msg)
            return error_msg
        
        # 检查是否有足够的有效数据点进行标定
        if len(self.objpoints) < 3:  # 需要至少3个有效图像
            error_msg = f"标定失败: 有效图像数量不足。只有 {len(self.objpoints)} 张有效图像，需要至少3张。"
            print(error_msg)
            return error_msg
        
        print(f"[DEBUG] 找到 {len(self.objpoints)} 张有效标定图像，开始执行标定...")
        
        # 检查图像尺寸
        h, w = self.calibration_images[0].shape
        image_size = (w, h)
        print(f"[DEBUG] 图像尺寸: {image_size}")
        
        # 检查数据质量
        print(f"[DEBUG] 数据质量检查:")
        for i, (obj, img) in enumerate(zip(self.objpoints, self.imgpoints)):
            print(f"[DEBUG]   - 第{i+1}组数据: 3D点形状={obj.shape}, 2D点形状={img.shape}")
            if obj.size == 0 or img.size == 0:
                print(f"[DEBUG]   - 警告: 第{i+1}组数据为空!")
            # 检查数据类型
            print(f"[DEBUG]   - 第{i+1}组数据类型: 3D点={obj.dtype}, 2D点={img.dtype}")
        
        print(f"[DEBUG] 标定前数据验证:")
        print(f"[DEBUG]   - 图像尺寸: {image_size}")
        print(f"[DEBUG]   - 3D点数量: {len(self.objpoints)} 组")
        print(f"[DEBUG]   - 2D点数量: {len(self.imgpoints)} 组")
        for i, (obj, img) in enumerate(zip(self.objpoints, self.imgpoints)):
            print(f"[DEBUG]   - 第{i+1}组: 3D点 {obj.shape}, 2D点 {img.shape}")
        
        # 执行标定
        try:
            if model == "pinhole":
                print(f"[DEBUG] 使用针孔模型进行标定")
                # 针孔模型：k1,k2,p1,p2,k3
                K = np.eye(3, dtype=np.float64)
                D = np.zeros((5, 1), dtype=np.float64)
                flags = 0
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
                
                print(f"[DEBUG] 准备针孔模型标定数据...")
                # 处理对象点和图像点以符合 calibrateCamera 函数的要求
                objp2 = []
                imgp2 = []
                for idx, (o, i) in enumerate(zip(self.objpoints, self.imgpoints)):
                    print(f"[DEBUG] 处理第 {idx+1} 组点对: obj shape={o.shape}, img shape={i.shape}")
                    # pinhole calibrateCamera 接受 obj (N,3) 或 (N,1,3)，这里统一成 (N,3)
                    if o.ndim == 3:
                        objp2.append(o.reshape(-1, 3).astype(np.float32))
                    else:
                        objp2.append(o.astype(np.float32))
                    # imgpoints 统一成 (N,1,2)
                    if i.ndim == 2:
                        imgp2.append(i.reshape(-1, 1, 2).astype(np.float32))
                    else:
                        imgp2.append(i.astype(np.float32))
                
                print(f"[DEBUG] 针孔标定数据准备完成:")
                print(f"[DEBUG]   - objp2 包含 {len(objp2)} 组数据")
                print(f"[DEBUG]   - imgp2 包含 {len(imgp2)} 组数据")
                for idx, (o, i) in enumerate(zip(objp2, imgp2)):
                    print(f"[DEBUG]   - 第{idx+1}组: obj {o.shape}, img {i.shape}")
                
                print(f"[DEBUG] 开始调用cv2.calibrateCamera，图像尺寸: {image_size}")
                ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                    objp2, imgp2, image_size, K, D, flags=flags, criteria=crit
                )
                print(f"[DEBUG] cv2.calibrateCamera 返回值: ret={ret}, K.shape={K.shape}, D.shape={D.shape}")
                print(f"[DEBUG] 旋转向量数量: {len(rvecs)}, 平移向量数量: {len(tvecs)}")
                
                mean_err = self._reprojection_error_pinhole(objp2, imgp2, rvecs, tvecs, K, D)
                print(f"[DEBUG] 针孔模型重投影误差: {mean_err:.4f} px")
            
            elif model == "fisheye":
                print(f"[DEBUG] 使用鱼眼模型进行标定")
                # 鱼眼模型：k1,k2,k3,k4
                objp_f = []
                imgp_f = []
                for idx, (o, i) in enumerate(zip(self.objpoints, self.imgpoints)):
                    print(f"[DEBUG] 处理第 {idx+1} 组鱼眼点对: obj shape={o.shape}, img shape={i.shape}")
                    if o.ndim == 2:
                        o = o.reshape(-1, 1, 3)
                    if i.ndim == 2:
                        i = i.reshape(-1, 1, 2)
                    objp_f.append(o.astype(np.float64))
                    imgp_f.append(i.astype(np.float64))

                print(f"[DEBUG] 鱼眼标定数据准备完成:")
                print(f"[DEBUG]   - objp_f 包含 {len(objp_f)} 组数据")
                print(f"[DEBUG]   - imgp_f 包含 {len(imgp_f)} 组数据")
                for idx, (o, i) in enumerate(zip(objp_f, imgp_f)):
                    print(f"[DEBUG]   - 第{idx+1}组: obj {o.shape}, img {i.shape}")
                
                K = np.eye(3, dtype=np.float64)
                D = np.zeros((4, 1), dtype=np.float64)
                flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
                
                print(f"[DEBUG] 开始调用cv2.fisheye.calibrate，图像尺寸: {image_size}")
                ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objp_f, imgp_f, image_size, K, D, None, None, flags=flags, criteria=crit
                )
                print(f"[DEBUG] cv2.fisheye.calibrate 返回值: ret={ret}, K.shape={K.shape}, D.shape={D.shape}")
                print(f"[DEBUG] 旋转向量数量: {len(rvecs)}, 平移向量数量: {len(tvecs)}")
                
                mean_err = self._reprojection_error_fisheye(objp_f, imgp_f, rvecs, tvecs, K, D)
                print(f"[DEBUG] 鱼眼模型重投影误差: {mean_err:.4f} px")
            else:
                raise ValueError("model 只能是 pinhole 或 fisheye")
            
            if ret and ret > 0:
                print(f"[DEBUG] 标定完成，误差: {ret}")
                print(f"[DEBUG] 内参矩阵K:\n{K}")
                print(f"[DEBUG] 畸变系数D:\n{D.flatten()}")
                self.camera_properties['calibration_matrix'] = K
                self.camera_properties['distortion_coeffs'] = D
                
                # 更新前端畸变矫正参数，以便在C++代码生成和实时显示中使用
                dist_coeffs = D.flatten()
                self.k1 = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
                self.k2 = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0
                self.p1 = float(dist_coeffs[2]) if len(dist_coeffs) > 2 else 0.0
                self.p2 = float(dist_coeffs[3]) if len(dist_coeffs) > 3 else 0.0
                self.k3 = float(dist_coeffs[4]) if len(dist_coeffs) > 4 else 0.0
                
                print(f"[DEBUG] 前端畸变参数已更新: k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3}")
                
                success_msg = f"{model}模型标定成功! 重投影误差: {mean_err:.4f} px. 内参矩阵和畸变系数已保存。"
                print(success_msg)
                return success_msg
            else:
                error_msg = f"标定失败: OpenCV标定函数返回值异常 (ret={ret})。"
                print(error_msg)
                return error_msg
        except Exception as e:
            import traceback
            print(f"[ERROR] 标定失败: 发生异常 - {str(e)}")
            print(f"[ERROR] 异常详细信息:\n{traceback.format_exc()}")
            error_msg = f"标定失败: 发生异常 - {str(e)}"
            print(error_msg)
            return error_msg
    
    def _reprojection_error_pinhole(self, objpoints, imgpoints, rvecs, tvecs, K, D):
        """计算针孔模型的重投影误差"""
        print(f"[DEBUG] 开始计算针孔模型重投影误差，共有 {len(objpoints)} 组数据")
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            print(f"[DEBUG] 处理第 {i+1} 组数据: obj={objpoints[i].shape}, img={imgpoints[i].shape}")
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            proj = proj.reshape(-1, 2)
            pts = imgpoints[i].reshape(-1, 2)
            err = np.linalg.norm(proj - pts, axis=1).mean()
            print(f"[DEBUG] 第 {i+1} 组数据误差: {err:.4f}")
            total_err += err * len(objpoints[i])
            total_pts += len(objpoints[i])
        mean_error = float(total_err / max(total_pts, 1))
        print(f"[DEBUG] 针孔模型总误差: {total_err}, 总点数: {total_pts}, 平均误差: {mean_error:.4f}")
        return mean_error
    
    def _reprojection_error_fisheye(self, objpoints, imgpoints, rvecs, tvecs, K, D):
        """计算鱼眼模型的重投影误差"""
        print(f"[DEBUG] 开始计算鱼眼模型重投影误差，共有 {len(objpoints)} 组数据")
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            print(f"[DEBUG] 处理第 {i+1} 组数据: obj={objpoints[i].shape}, img={imgpoints[i].shape}")
            proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            proj = proj.reshape(-1, 2)
            pts = imgpoints[i].reshape(-1, 2)
            err = np.linalg.norm(proj - pts, axis=1).mean()
            print(f"[DEBUG] 第 {i+1} 组数据误差: {err:.4f}")
            total_err += err * len(objpoints[i])
            total_pts += len(objpoints[i])
        mean_error = float(total_err / max(total_pts, 1))
        print(f"[DEBUG] 鱼眼模型总误差: {total_err}, 总点数: {total_pts}, 平均误差: {mean_error:.4f}")
        return mean_error
    
    def generate_cpp_code(self):
        """生成C++代码 - 摄像头参数设置代码，包含畸变矫正参数"""
        # 根据当前摄像头参数生成C++代码
        cpp_code = f"""#include <opencv2/opencv.hpp>
#include <iostream>

int main() {{
    // 创建摄像头对象
    cv::VideoCapture cap({self.current_camera});
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {{
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }}
    
    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, {self.camera_properties['resolution_width']});      // 分辨率宽度
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, {self.camera_properties['resolution_height']});    // 分辨率高度
    cap.set(cv2.CAP_PROP_FPS, {self.camera_properties['fps']});                           // 帧率
    
    // 曝光设置
    """
        
        if self.camera_properties['exposure_mode'] == 'auto':
            cpp_code += "    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75);  // 自动曝光\n"
        else:
            cpp_code += "    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // 手动曝光\n"
            cpp_code += f"    cap.set(cv::CAP_PROP_EXPOSURE, {self.camera_properties['exposure_value']});   // 曝光值\n"
        
        cpp_code += """
    // 摄像头内参矩阵 (标定结果)
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
    
    // 畸变系数 (k1, k2, p1, p2, k3)
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
        {}, {}, {}, {}, {}  // k1, k2, p1, p2, k3
    );
    
    // 如果已执行标定，使用标定结果
""".format(self.k1, self.k2, self.p1, self.p2, self.k3)
        
        if self.camera_properties['calibration_matrix'] is not None and self.camera_properties['distortion_coeffs'] is not None:
            cam_mat = self.camera_properties['calibration_matrix']
            dist_coeffs = self.camera_properties['distortion_coeffs']
            cpp_code += f"""    cameraMatrix = (cv::Mat_<double>(3, 3) << 
        {cam_mat[0,0]:.6f}, {cam_mat[0,1]:.6f}, {cam_mat[0,2]:.6f},
        {cam_mat[1,0]:.6f}, {cam_mat[1,1]:.6f}, {cam_mat[1,2]:.6f},
        {cam_mat[2,0]:.6f}, {cam_mat[2,1]:.6f}, {cam_mat[2,2]:.6f}
    );
    
    distCoeffs = (cv::Mat_<double>(1, 5) << 
        {dist_coeffs[0,0]:.6f}, 
        {dist_coeffs[1,0]:.6f}, 
        {dist_coeffs[2,0]:.6f}, 
        {dist_coeffs[3,0]:.6f}, 
        {dist_coeffs[4,0]:.6f}
    );
    
"""
        else:
            cpp_code += f"""    // 注意: 未执行标定，使用默认参数
    // 请执行标定以获得准确的内参矩阵和畸变系数
"""
        
        cpp_code += """    // 读取并显示摄像头画面
    cv::Mat frame;
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            std::cerr << "无法接收帧，退出" << std::endl;
            break;
        }
        
        // 应用畸变矫正
        cv::Mat undistortedFrame;
        cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);
        
        // 显示原始画面和矫正后画面
        cv::imshow("Original", frame);
        cv::imshow("Undistorted", undistortedFrame);
        
        // 按 'q' 键退出
        if (cv2::waitKey(1) == 'q') {
            break;
        }
    }
    
    // 释放摄像头资源
    cap.release();
    cv2::destroyAllWindows();
    
    return 0;
}"""
        return cpp_code
    
    def get_charuco_board_image(self):
        """获取ChArUco标定板图像用于显示"""
        # 生成标定板图像
        board_image = self.board.generateImage((600, 800))
        # 转换为RGB格式用于Gradio显示
        board_image_rgb = cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
        return board_image_rgb
    
    def set_board_type(self, board_type, **kwargs):
        """设置标定板类型"""
        self.board_type = board_type
        if board_type == "chessboard":
            if 'chessboard_size' in kwargs:
                self.chessboard_size = kwargs['chessboard_size']  # (cols, rows)
            if 'square_size' in kwargs:
                self.square_size = kwargs['square_size']  # mm
        elif board_type == "charuco":
            # 更新ChArUco标定板参数
            if all(k in kwargs for k in ('squares_x', 'squares_y', 'square_len', 'marker_len', 'dict_name')):
                dict_id = getattr(cv2.aruco, kwargs['dict_name'])
                self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                self.board = cv2.aruco.CharucoBoard(
                    (kwargs['squares_x'], kwargs['squares_y']),
                    float(kwargs['square_len']),
                    float(kwargs['marker_len']),
                    self.dictionary
                )
                if 'legacy_pattern' in kwargs and kwargs['legacy_pattern'] and hasattr(self.board, "setLegacyPattern"):
                    self.board.setLegacyPattern(True)
        return f"Board type set to {board_type}"
    
    def get_board_image(self, width=600, height=800):
        """获取标定板图像用于显示"""
        if self.board_type == "chessboard":
            # 生成棋盘格图像
            cols, rows = self.chessboard_size
            square_size = 10  # 设置小的方格尺寸以适应显示
            img_size = (width, height)
            
            # 计算棋盘格实际大小
            board_w = cols * square_size
            board_h = rows * square_size
            
            # 创建图像
            img = np.zeros((board_h, board_w, 3), dtype=np.uint8)
            
            # 绘制棋盘格
            for y in range(rows):
                for x in range(cols):
                    if (x + y) % 2 == 0:
                        cv2.rectangle(img, (x * square_size, y * square_size), 
                                    ((x + 1) * square_size, (y + 1) * square_size), 
                                    (255, 255, 255), -1)
                    else:
                        cv2.rectangle(img, (x * square_size, y * square_size), 
                                    ((x + 1) * square_size, (y + 1) * square_size), 
                                    (0, 0, 0), -1)
            
            # 缩放到指定尺寸
            img_resized = cv2.resize(img, img_size)
            return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:  # charuco
            # 生成ChArUco标定板图像
            board_image = self.board.generateImage((width, height), marginSize=20, borderBits=1)
            # 转换为RGB格式用于Gradio显示
            return cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
    
    def get_charuco_board_image(self):
        """获取ChArUco标定板图像用于显示"""
        # 为了向后兼容，调用通用方法
        return self.get_board_image()
    
    def close_camera(self):
        """关闭摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None