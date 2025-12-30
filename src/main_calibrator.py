"""重构后的主标定器类"""
from .camera import CameraManager
from .board import BoardManager
from .calibration import CalibrationCore
from .utils import ensure_dir, save_yaml


class CameraCalibrator:
    """重构后的摄像头标定器主类"""
    def __init__(self):
        # 初始化各模块
        self.camera_manager = CameraManager()
        self.board_manager = BoardManager()
        self.calibration_core = CalibrationCore()
        
        # 初始化矫正参数
        self.k1 = 0.0  # 径向畸变系数
        self.k2 = 0.0  # 径向畸变系数
        self.p1 = 0.0  # 切向畸变系数
        self.p2 = 0.0  # 切向畸变系数
        self.k3 = 0.0  # 径向畸变系数

    # 摄像头管理方法
    def get_camera_list(self):
        """获取可用摄像头列表"""
        return self.camera_manager.get_camera_list()

    def start_camera(self, camera_index):
        """启动摄像头"""
        return self.camera_manager.start_camera(camera_index)

    def update_camera_params(self, resolution_width, resolution_height, fps, exposure_mode, exposure_value):
        """更新摄像头参数"""
        return self.camera_manager.update_camera_params(
            resolution_width, resolution_height, fps, exposure_mode, exposure_value
        )

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
        # 合并标定数据和摄像头属性
        calibration_data = {
            'calibration_matrix': self.camera_manager.camera_properties.get('calibration_matrix'),
            'distortion_coeffs': self.camera_manager.camera_properties.get('distortion_coeffs')
        }
        return self.camera_manager.get_camera_frame(self.board_manager, calibration_data)

    def close_camera(self):
        """关闭摄像头"""
        self.camera_manager.close_camera()

    # 标定板管理方法
    def set_board_type(self, board_type, **kwargs):
        """设置标定板类型"""
        return self.board_manager.set_board_type(board_type, **kwargs)

    def get_board_image(self, width=600, height=800):
        """获取标定板图像用于显示"""
        return self.board_manager.get_board_image(width, height)

    def get_charuco_board_image(self):
        """获取ChArUco标定板图像用于显示"""
        # 为了向后兼容，调用通用方法
        return self.get_board_image()

    # 标定相关方法
    def capture_calibration_image(self):
        """捕获用于标定的图像"""
        if self.camera_manager.cap is not None and self.camera_manager.cap.isOpened():
            ret, frame = self.camera_manager.cap.read()
            if ret:
                return self.calibration_core.capture_calibration_image(frame, self.board_manager)
            else:
                return "Failed to capture image"
        else:
            return "Camera is not opened"

    def calibrate_camera(self, model="pinhole"):
        """执行摄像头标定"""
        result_msg, calibration_result = self.calibration_core.calibrate_camera(
            model, self.camera_manager.camera_properties
        )
        
        if calibration_result:
            # 更新摄像头属性
            self.camera_manager.camera_properties['calibration_matrix'] = calibration_result['calibration_matrix']
            self.camera_manager.camera_properties['distortion_coeffs'] = calibration_result['distortion_coeffs']
            # 更新主类的畸变参数
            self.k1 = self.calibration_core.k1
            self.k2 = self.calibration_core.k2
            self.p1 = self.calibration_core.p1
            self.p2 = self.calibration_core.p2
            self.k3 = self.calibration_core.k3
            
        return result_msg

    def generate_cpp_code(self):
        """生成C++代码 - 摄像头参数设置代码，包含畸变矫正参数"""
        distortion_params = {
            'k1': self.k1,
            'k2': self.k2,
            'p1': self.p1,
            'p2': self.p2,
            'k3': self.k3
        }
        return self.calibration_core.generate_cpp_code(
            self.camera_manager.camera_properties, 
            distortion_params
        )

    def save_calibration(self, out_path, model="pinhole", extra=None):
        """保存标定结果到YAML文件"""
        if (self.camera_manager.camera_properties['calibration_matrix'] is None or 
            self.camera_manager.camera_properties['distortion_coeffs'] is None):
            return "没有标定数据可保存"
        
        print(f"[DEBUG] 开始保存标定结果到: {out_path}")
        print(f"[DEBUG] 模型类型: {model}")
        
        print(f"[DEBUG] 标定矩阵信息:")
        print(f"[DEBUG]   - 内参矩阵K:\n{self.camera_manager.camera_properties['calibration_matrix']}")
        print(f"[DEBUG]   - 畸变系数D:\n{self.camera_manager.camera_properties['distortion_coeffs'].flatten()}")
        print(f"[DEBUG]   - 图像尺寸: {self.camera_manager.camera_properties['resolution_width']}x{self.camera_manager.camera_properties['resolution_height']}")
        
        # 使用工具函数保存
        image_size = (
            self.camera_manager.camera_properties['resolution_width'],
            self.camera_manager.camera_properties['resolution_height']
        )
        save_yaml(
            out_path, 
            image_size, 
            self.camera_manager.camera_properties['calibration_matrix'],
            self.camera_manager.camera_properties['distortion_coeffs'],
            model,
            extra
        )
        
        print(f"[DEBUG] 标定结果已保存到: {out_path}")
        return f"标定结果已保存到: {out_path}"