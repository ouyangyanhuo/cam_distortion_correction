"""标定核心算法模块"""
import cv2
import numpy as np
from .utils import reprojection_error_pinhole, reprojection_error_fisheye
from .detection import detect_chessboard, detect_charuco


class CalibrationCore:
    """标定核心算法类"""
    def __init__(self):
        # 标定相关
        self.calibration_images = []
        self.objpoints = []  # 3D点
        self.imgpoints = []  # 2D点
        
        # 矫正参数
        self.k1 = 0.0  # 径向畸变系数
        self.k2 = 0.0  # 径向畸变系数
        self.p1 = 0.0  # 切向畸变系数
        self.p2 = 0.0  # 切向畸变系数
        self.k3 = 0.0  # 径向畸变系数

    def capture_calibration_image(self, frame, board_manager):
        """捕获用于标定的图像"""
        print(f"[DEBUG] 开始捕获标定图像，当前标定板类型: {board_manager.board_type}")
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"[DEBUG] 成功读取图像，图像尺寸: {frame.shape}")
            
            if board_manager.board_type == "chessboard":
                print(f"[DEBUG] 检测棋盘格角点，尺寸: {board_manager.chessboard_size}")
                # 检测棋盘格角点
                corners = detect_chessboard(gray, board_manager.chessboard_size)
                if corners is not None and len(corners) >= 4:
                    print(f"[DEBUG] 检测到 {len(corners)} 个棋盘格角点")
                    # 为棋盘格创建对象点
                    objp = np.zeros((board_manager.chessboard_size[0] * board_manager.chessboard_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:board_manager.chessboard_size[0], 0:board_manager.chessboard_size[1]].T.reshape(-1, 2)
                    objp *= float(board_manager.square_size)
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
                charuco_corners, charuco_ids = detect_charuco(gray, board_manager.board, board_manager.dictionary)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 6:
                    print(f"[DEBUG] 检测到 {len(charuco_corners)} 个 ChArUco 角点, {len(charuco_ids)} 个 ID")
                    # 获取3D棋盘格角点
                    chess_corners_3d = board_manager.board.getChessboardCorners()
                    
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

    def calibrate_camera(self, model="pinhole", camera_properties=None):
        """执行摄像头标定
        
        Args:
            model (str): 相机模型, 'pinhole' 或 'fisheye'
            camera_properties (dict): 摄像头属性，包含分辨率等信息
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
                
                mean_err = reprojection_error_pinhole(objp2, imgp2, rvecs, tvecs, K, D)
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
                
                mean_err = reprojection_error_fisheye(objp_f, imgp_f, rvecs, tvecs, K, D)
                print(f"[DEBUG] 鱼眼模型重投影误差: {mean_err:.4f} px")
            else:
                raise ValueError("model 只能是 pinhole 或 fisheye")
            
            if ret and ret > 0:
                print(f"[DEBUG] 标定完成，误差: {ret}")
                print(f"[DEBUG] 内参矩阵K:\n{K}")
                print(f"[DEBUG] 畸变系数D:\n{D.flatten()}")
                
                # 更新前端畸变参数，以便在C++代码生成和实时显示中使用
                dist_coeffs = D.flatten()
                self.k1 = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
                self.k2 = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0
                self.p1 = float(dist_coeffs[2]) if len(dist_coeffs) > 2 else 0.0
                self.p2 = float(dist_coeffs[3]) if len(dist_coeffs) > 3 else 0.0
                self.k3 = float(dist_coeffs[4]) if len(dist_coeffs) > 4 else 0.0
                
                print(f"[DEBUG] 前端畸变参数已更新: k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3}")
                
                result = {
                    'calibration_matrix': K,
                    'distortion_coeffs': D,
                    'reprojection_error': mean_err,
                    'model': model
                }
                
                success_msg = f"{model}模型标定成功! 重投影误差: {mean_err:.4f} px. 内参矩阵和畸变系数已保存。"
                print(success_msg)
                return success_msg, result
            else:
                error_msg = f"标定失败: OpenCV标定函数返回值异常 (ret={ret})。"
                print(error_msg)
                return error_msg, None
        except Exception as e:
            import traceback
            error_msg = f"标定失败: 发生异常 - {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 异常详细信息:\n{traceback.format_exc()}")
            return error_msg, None

    def generate_cpp_code(self, camera_properties, distortion_params=None):
        """生成C++代码 - 摄像头参数设置代码，包含畸变矫正参数"""
        if distortion_params is None:
            distortion_params = {'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0}
        
        # 根据当前摄像头参数生成C++代码
        cpp_code = f"""#include <opencv2/opencv.hpp>
#include <iostream>

int main() {{
    // 创建摄像头对象
    cv::VideoCapture cap(0);
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {{
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }}
    
    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, {camera_properties.get('resolution_width', 320)});      // 分辨率宽度
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, {camera_properties.get('resolution_height', 240)});    // 分辨率高度
    cap.set(cv2.CAP_PROP_FPS, {camera_properties.get('fps', 30)});                           // 帧率
    
    // 曝光设置
    """
        
        if camera_properties.get('exposure_mode', 'auto') == 'auto':
            cpp_code += "    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75);  // 自动曝光\n"
        else:
            cpp_code += "    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // 手动曝光\n"
            cpp_code += f"    cap.set(cv::CAP_PROP_EXPOSURE, {camera_properties.get('exposure_value', -6)});   // 曝光值\n"
        
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
""".format(distortion_params['k1'], distortion_params['k2'], distortion_params['p1'], distortion_params['p2'], distortion_params['k3'])
        
        if camera_properties.get('calibration_matrix') is not None and camera_properties.get('distortion_coeffs') is not None:
            cam_mat = camera_properties['calibration_matrix']
            dist_coeffs = camera_properties['distortion_coeffs']
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