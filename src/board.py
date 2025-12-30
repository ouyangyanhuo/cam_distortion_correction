"""标定板管理模块"""
import cv2
import numpy as np
from .detection import detect_chessboard, detect_charuco


class BoardManager:
    """标定板管理类"""
    def __init__(self, board_type="charuco", **kwargs):
        self.board_type = board_type
        self.chessboard_size = (9, 6)  # 棋盘格内角点数量 (cols, rows)
        self.square_size = 25.0  # 方格大小 (mm)
        
        # 初始化ChArUco标定板
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.board = cv2.aruco.CharucoBoard((7, 10), 25.0, 17.5, self.dictionary)
        
        # 更新参数（如果提供）
        if 'chessboard_size' in kwargs:
            self.chessboard_size = kwargs['chessboard_size']
        if 'square_size' in kwargs:
            self.square_size = kwargs['square_size']
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

    def detect_board_corners(self, gray):
        """检测标定板角点"""
        if self.board_type == "chessboard":
            corners = detect_chessboard(gray, self.chessboard_size)
            return corners, None  # 棋盘格只返回角点
        else:  # charuco
            charuco_corners, charuco_ids = detect_charuco(gray, self.board, self.dictionary)
            return charuco_corners, charuco_ids

    def get_board_image(self, width=600, height=800):
        """获取标定板图像用于显示"""
        if self.board_type == "chessboard":
            # 生成棋盘格图像
            cols, rows = self.chessboard_size
            square_size = 10  # 设置小的方格尺寸以适应显示
            
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
            img_resized = cv2.resize(img, (width, height))
            return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:  # charuco
            # 生成ChArUco标定板图像
            board_image = self.board.generateImage((width, height), marginSize=20, borderBits=1)
            # 转换为RGB格式用于显示
            return cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)